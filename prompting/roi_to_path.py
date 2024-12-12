import dataclasses
from copy import deepcopy

from jaxvox import VoxGrid
from jaxvox.utils.planning import plan_many_robots, raypaths
import jax.numpy as jnp
import open3d as o3d
import numpy as np
from rocobench import RobotState


@dataclasses.dataclass
class RobotAction:
    robot_name: str
    action: str
    robot_name_pos: jnp.ndarray
    goal_name: str
    goal_pos: jnp.ndarray
    path: jnp.ndarray = None

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)
    @property
    def needs_completion(self):
        if self.goal_name is False:
            return False
        if self.path is None:
            return True
        return False

    def complete(self, path):
        return self.replace(path=path)

    def to_sentence(self):
        return f"NAME {self.robot_name} ACTION {self.action} {self.goal_name} PATH {[tuple(list(np.array(x))) for x in self.path]}"


def _parse_execute(execute_message, pos_dict):
    _, execute_message = execute_message.split("EXECUTE")
    assert len(execute_message) > 1 and (" alice " in execute_message.lower() and " bob " in execute_message.lower())

    def handle_robot(robot_sentence):
        robot_name = robot_sentence.split("NAME ")[-1].split("ACTION")[0].strip()

        robot_action = " ".join(robot_sentence.split("ACTION ")[-1].split(" ")[:-1]).strip()
        goal = robot_sentence.split("ACTION ")[-1].split(" ")[-1].strip()

        VALID_ACTIONS = ['PICK', 'PLACE', 'PUT', 'OPEN', 'SWEEP', 'DUMP', 'MOVE']
        NOOP_ACTIONS = ['WAIT']

        R_A = RobotAction(robot_name=robot_name, action=robot_action, robot_name_pos=pos_dict[robot_name.lower()], goal_name=None, goal_pos=None)

        FOUND_ACTION = False
        for VA in VALID_ACTIONS:
            if R_A.action.startswith(VA):
                FOUND_ACTION = True
                R_A = R_A.replace(
                    goal_name=goal,
                    goal_pos=pos_dict[goal.lower()]
                )
                break
        if not FOUND_ACTION:
            for NA in NOOP_ACTIONS:
                if R_A.action.startswith(NA):
                    FOUND_ACTION = True
        if not FOUND_ACTION:
            raise AssertionError(f"Could not parse robot action {{{robot_action}}}")

        return R_A

    alice_sentence, bob_sentence = execute_message.strip().split("\n")
    assert alice_sentence.lower().startswith("name alice") and bob_sentence.lower().startswith("name bob")

    alice_RA = handle_robot(alice_sentence)
    bob_RA = handle_robot(bob_sentence)

    to_plan = [alice_RA, bob_RA]
    return to_plan

def _get_posdict(env, obs):
    """ Display the plan in the open3d viewer """
    obj_desp = env.get_object_desp(obs)

    def line_by_line(line):
        line = line.split("),")[0]+",)"
        objname = line.split(": (")[0]
        objpos = "(" + line.split(": (")[1]
        objpos = objpos.replace("(", '').replace(",)", ")").replace(")", "")

        objpos = np.array(list(map(float, objpos.split(","))))
        return objname, objpos
    name2pos = [line_by_line(line) for line in obj_desp.split("\n")]
    objdict = {k:v for k,v in name2pos}

    for robot_name, agent_name in env.robot_name_map.items():
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos
        objdict[agent_name] = np.array([x, y, z])
        objdict[agent_name.lower()] = np.array([x, y, z])
    return objdict

    env = deepcopy(env)
    env.physics.data.qpos[:] = env.physics.data.qpos[:].copy()
    env.physics.forward()
    env.render_point_cloud = True
    #obs = env.get_obs()

    poses = {}
    for k, v in vars(obs).items():
        if isinstance(v, RobotState):
            poses[f"{k}_pos"] = v.ee_xpos
            poses[env.robot_name_map[k]] = v.ee_xpos
        elif k == "objects":
            for objname, objval in v.items():
                if objname == "bin":
                    continue

                if objname == "table_top":
                    pos = env.physics.data.site(f"{objname}").xpos
                    if objname == "table":
                        pos[-1] += 0.15
                else:
                    pos = env.physics.data.site(f"{objname}_top").xpos
                    pos[-1] += 0.05

                poses[objname] = pos
        elif k == "bin_slot_xposes":
            poses.update(v)

    poses = {k.lower(): tuple(list(v)) for k, v in poses.items()}
    return poses

def execute_completion(execute_message, env, obs, voxel_size=0.02):
    # 0. get pose dict
    env = deepcopy(env)
    env.render_point_cloud = True
    obs = env.get_obs()

    pos_dict = _get_posdict(env, obs)

    env = deepcopy(env)
    def disable_robot(env, robot):
        # Get the body ID for the robot
        robot_body_id = env.physics.model.name2id(robot, 'body')

        # Retrieve all geom IDs associated with this body
        geom_ids = [
            i for i in range(env.physics.model.ngeom)
            if env.physics.model.geom_bodyid[i] == robot_body_id
        ]

        def get_all_geoms_for_body(physics, root_body_id):
            geom_ids = []

            # Recursive function to collect all geom IDs
            def collect_geom_ids(body_id):
                for i in range(physics.model.nbody):
                    if physics.model.body_parentid[i] == body_id:
                        collect_geom_ids(i)  # Recursively collect child body geoms
                geom_ids.extend([
                    i for i in range(physics.model.ngeom)
                    if physics.model.geom_bodyid[i] == body_id
                ])

            collect_geom_ids(root_body_id)
            return geom_ids
        geom_ids = get_all_geoms_for_body(env.physics, robot_body_id)


        # Retrieve the names of these geoms
        for geom_id in geom_ids:

            env.physics.model.geom_rgba[geom_id] = [0, 0, 0, 0]  # Make it invisible
            env.physics.model.geom_size[geom_id] = [0, 0, 0]  # Set its size to zero
            # get already_existing pos that we know is valid
            for k, v in pos_dict.items():
                if k.lower() in ["alice", "bob"]:
                    continue
                env.physics.model.geom_pos[geom_id] = v
                break

    for robot in env.robot_name_map.keys():
        disable_robot(env, robot)

    obs = env.get_obs()

    # 1. parse
    alice_ra, bob_ra = _parse_execute(execute_message, pos_dict)

    # 2.
    if (not alice_ra.needs_completion) and (not bob_ra.needs_completion):
        return execute_message

    pcd = obs.scene.to_open3d()
    voxgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxgrid, attrmanager = VoxGrid.from_open3d(voxgrid, import_attrs=True,
                                               return_attrmanager=True)
    attrmanager.set_default_value((255, 0, 0))
    voxgrid.display_as_o3d(attrmanager)

    if alice_ra.needs_completion and bob_ra.needs_completion:
        p1, p2 = plan_many_robots(voxgrid, jnp.array([alice_ra.robot_name_pos, bob_ra.robot_name_pos]), jnp.array([alice_ra.goal_pos, bob_ra.goal_pos]), batch_size=10, dist_tol=2, radius_tol=4, expand_size=2)

        alice_ra = alice_ra.complete(p1)
        bob_ra = bob_ra.complete(p2)

        if True:
            voxgrid = raypaths(voxgrid, alice_ra.path, 2, 1)
            voxgrid = raypaths(voxgrid, bob_ra.path, 2, 20)

            voxgrid.display_as_o3d(attrmanager)

        return f"""
EXECUTE
{alice_ra.to_sentence()}
{bob_ra.to_sentence()}
"""
        i = 0
    raise NotImplementedError()




