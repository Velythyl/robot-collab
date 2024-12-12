import os
import json
import pickle
from time import sleep

import openai
import numpy as np
from datetime import datetime
from os.path import join
from typing import List, Tuple, Dict, Union, Optional, Any

from rocobench.subtask_plan import LLMPathPlan
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs import MujocoSimEnv, EnvState
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .roi_to_path import execute_completion

assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
OPENAI_KEY = str(json.load(open("openai_key.json")))
openai.api_key = OPENAI_KEY

PATH_PLAN_INSTRUCTION = """
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""


class DialogPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent
    """

    def __init__(
            self,
            env: MujocoSimEnv,
            parser: LLMResponseParser,
            feedback_manager: FeedbackManager,
            max_tokens: int = 512,
            debug_mode: bool = False,
            use_waypoints: bool = False,
            robot_name_map: Dict[str, str] = {"panda": "Bob"},
            num_replans: int = 3,
            max_calls_per_round: int = 10,
            use_history: bool = True,
            use_feedback: bool = True,
            temperature: float = 0,
            llm_source: str = "gpt-4"
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        #env.render_point_cloud = True
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = []
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round
        self.temperature = temperature
        self.llm_source = llm_source
        assert llm_source in ["gpt-4", "gpt-3.5-turbo",
                              "claude", "olmo2-15b"], f"llm_source invalid, got {llm_source}"

    def compose_system_prompt(
            self,
            obs: EnvState,
            agent_name: str,
            chat_history: List = [],  # chat from previous replan rounds
            current_chat: List = [],  # chat from current round, this comes AFTER env feedback
            feedback_history: List = []
    ) -> str:
        #action_desp = self.env.get_action_prompt()
        #if self.use_waypoints:
        #    action_desp += PATH_PLAN_INSTRUCTION
        #agent_prompt = self.env.get_agent_prompt(obs, agent_name)

        #round_history = self.get_round_history() if self.use_history else ""

        #execute_feedback = ""
        #if len(self.failed_plans) > 0:
        #    execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
        #    execute_feedback += "\n".join(self.failed_plans) + "\n"

        #chat_history = "[Previous Chat]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""

        #system_prompt = f"{action_desp}\n{round_history}\n{execute_feedback}{agent_prompt}\n{chat_history}\n"

        system_prompt = f"""
[TASK]
Two robot arms, Alice and Bob, stand on different sides of a table, and together must put all the grocery items into a bin. They must avoid collisions, both robot-robot and robot-object.

[WORLD STATE]
{self.env.get_object_desp(obs)}
{self.env.get_robots_desp(obs)}
{self.env.get_table_constraint_desp(obs)}

[ACTION]
1) PICK <obj>
2) PLACE <obj> <bin slot>
Obviously, cannot PLACE objects that are not held by the gripper. Need to PICK first.

[DISCUSSION]
The robots use their supercomputers to discuss the task and what actions they want to take. 
First, they clearly state “I am Alice” or “I am Bob”. 
Second, they read the [CHAT WINDOW] and give feedback to the other robot, if need be. For example, if Alice picked an object that is too close to Bob's best choice, Bob might tell Alice to pick a different object.
Third, they state what their action will be. 
They can only choose one action each. Once they are both confident in what they want to do, the OVERSEER will take over and output an EXECUTE message. 
The OVERSEER only intervenes when both Alice and Bob have had a chance to speak. All robots must select an action. They cannot target the same object.

[EXECUTE]
EXECUTE
NAME Alice ACTION <ACTION>
NAME Bob ACTION <ACTION>
Example:
EXECUTE
NAME Alice ACTION PICK controller 
NAME Bob ACTION PICK dog

[CHAT FORMAT]
Alice: I am Alice. I will ...
Bob: I am Bob. No, you should ... Because ...
...
OVERSEER:
EXECUTE
NAME Alice ACTION <ACTION 1 OR 2>
NAME Bob ACTION <ACTION 1 OR 2>

Remember, you can only generate ONE message, either Alice, Bob, or OVERSEER. Stop generating as soon as you output a message.
Remember that the OVERSEER is able to take over and send the EXECUTE message once both Alice and Bob have decided what they want to do.
Remember, the OVERSEER is hyper intelligent and always intervenes when both Bob and Alice agree on a plan.
The OVERSEER does not intervene prematurely. It makes sure that both Bob and Alice have chosen an ACTION.

---

"""

        if self.use_feedback and len(feedback_history) > 0:
            system_prompt += "\n".join(feedback_history)
        else:
            system_prompt += "[Environment feedback]\nNone at present"

        system_prompt += """

[CHAT WINDOW]
"""
        if len(current_chat) > 0:
            system_prompt += "\n".join(current_chat) + "\n"

        system_prompt += f"\nWaiting for {agent_name}'s message..."

        return system_prompt

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def prompt_one_round(self, obs: EnvState, save_path: str = ""):
        plan_feedbacks = []
        chat_history = []
        for i in range(self.num_replans):
            final_agent, final_response, agent_responses = self.prompt_one_dialog_round(
                obs,
                chat_history,
                plan_feedbacks,
                replan_idx=i,
                save_path=save_path,
            )
            chat_history += agent_responses

            final_response = execute_completion(final_response, self.env, obs)

            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, final_response)

            curr_feedback = "None"
            if not parse_succ:
                curr_feedback = f"""
This previous response from [{final_agent}] failed to parse!: '{final_response}'
{parsed_str} Re-format to strictly follow [Action Output Instruction]!"""
                ready_to_execute = False

            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans):
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (final_response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))

            if ready_to_execute:
                break
            else:
                print(curr_feedback)
        self.latest_chat_history = chat_history
        return ready_to_execute, llm_plans, plan_feedbacks, chat_history

    def prompt_one_dialog_round(
            self,
            obs,
            chat_history,
            feedback_history,
            replan_idx=0,
            save_path='data/',
    ):
        """
        keep prompting until an EXECUTE is outputted or max_calls_per_round is reached
        """

        agent_responses = []
        usages = []
        dialog_done = False
        num_responses = {agent: 0 for agent in self.robot_agent_names}
        n_calls = 0

        while n_calls < self.max_calls_per_round:
            for agent_name in self.robot_agent_names:
                system_prompt = self.compose_system_prompt(
                    obs,
                    agent_name,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,
                )

                agent_prompt = f"You are {agent_name}, your response is:"
                if n_calls == self.max_calls_per_round - 1:
                    agent_prompt = f"""
    You are {agent_name}, this is the last call, you must end your response by incoporating all previous discussions and output the best plan via EXECUTE. 
    Your response is:
                    """
                response, usage = self.query_once(
                    system_prompt,
                    user_prompt=agent_prompt,
                    max_query=3,
                )

                tosave = [
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": agent_prompt,
                    },
                    {
                        "sender": agent_name,
                        "message": response,
                    },
                    usage,
                ]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                fname = f'{save_path}/replan{replan_idx}_call{n_calls}_agent{agent_name}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))

                num_responses[agent_name] += 1
                # strip all the repeated \n and blank spaces in response:
                pruned_response = response.strip()
                # pruned_response = pruned_response.replace("\n", " ")
                agent_responses.append(
                    f"[{agent_name}]:\n{pruned_response}"
                )
                usages.append(usage)
                n_calls += 1
                if 'EXECUTE' in response:
                    if replan_idx > 0 or all([v > 0 for v in num_responses.values()]):
                        dialog_done = True
                        break

                if self.debug_mode:
                    dialog_done = True
                    break

            if dialog_done:
                break

        # response = "\n".join(response.split("EXECUTE")[1:])
        # print(response)



        return agent_name, response, agent_responses

    def query_once(self, system_prompt, user_prompt, max_query):
        response = None
        usage = None
        # print('======= system prompt ======= \n ', system_prompt)
        # print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode:
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()

#        return """
#EXECUTE
#NAME Alice ACTION PICK apple PATH [(0.29, 0.06, 0.51), (0.32, 0.12, 0.51), (0.35, 0.18, 0.51), (0.25, 0.41, 0.36)]
#NAME Bob ACTION PICK soda_can PATH [(0.35, 1.05, 0.62), (0.40, 1.00, 0.62), (0.45, 0.95, 0.62), (0.65, 0.41, 0.36)]""", {}

        if sum([int(x in self.llm_source) for x in ["chat", "claude"]]) == 0:
            from prompting.llm_wrapper import llm_autocomplete
            return llm_autocomplete(max_query, self.llm_source, system_prompt, user_prompt, self.max_tokens,
                                    self.temperature)

        for n in range(max_query):
            sleep(1)
            print('querying {}th time'.format(n))
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm_source,
                    messages=[
                        # {"role": "user", "content": ""},
                        {"role": "system", "content": system_prompt + user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                usage = response['usage']
                response = response['choices'][0]['message']["content"]
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except:
                print("API error, try again")
            sleep(5)
            continue
        # breakpoint()
        return response, usage

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success:
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            chats = "\n".join(self.latest_chat_history)
            self.round_history.append(
                f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = []
        self.latest_chat_history = []