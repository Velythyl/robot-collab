
import os
import json
import openai
from barebonesllmchat.terminal.interface import ChatbotClient

OPENAI_KEY = None
def _query_chatgpt(model, system_prompt, user_prompt, max_tokens, temperature):
    global OPENAI_KEY
    if OPENAI_KEY is None:
        assert os.path.exists(
            "openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
        OPENAI_KEY = str(json.load(open("openai_key.json")))
        openai.api_key = OPENAI_KEY

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            # {"role": "user", "content": ""},
            {"role": "system", "content": system_prompt + user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = response['usage']
    response = response['choices'][0]['message']["content"]
    return response, usage

barebonesllmchat_CLIENT = None
def _query_barebonesllmchat(model, system_prompt, user_prompt, max_tokens, temperature):

    global barebonesllmchat_CLIENT
    if barebonesllmchat_CLIENT is None:
        barebonesllmchat_CLIENT = ChatbotClient("http://127.0.0.1:5000", "your_api_key", use_websocket=True)

    from barebonesllmchat.common.chat_history import ChatHistory, CHAT_ROLE
    #system_prompt = f"{system_prompt}"

    PREFIX = """[Overview]
    You are an agent that is part of a robot task. The two robots must collaborate and communicate with each other.
    Below, you will find a list of rules for the interactions. Please follow those rules very closely.
    You'll notice that EXECUTE indicates a final plan. You can only output EXECUTE if you have preliminary plans for both robots.
    When you opt to EXECUTE, you must copy-paste the two plants in the following format:
    
    EXECUTE
    NAME <name> ACTION <action> <object> PATH <list of coords>
    NAME <name> ACTION <action> <object> PATH <list of coords>
    
    Notice that there is a plan for both robots.
    """
    #MIDFIX = """\n[Overview]
    #You are an agent that is part of a robot task. The two robots must collaborate and communicate with each other.
    #Below, you will find a list of rules for the interactions. Please follow those rules very closely.
    #You'll notice that EXECUTE indicates a final plan. You can only output EXECUTE if you have preliminary plans for both robots.
    #Also, make sure you PICK an object before PLACEing it.
    #Again, do not output EXECUTE if you do not have plans for both robots. EXECUTE should also include plans for both robots.
    #\n\n"""
    history = ChatHistory().add( CHAT_ROLE.SYSTEM, PREFIX +"\n"+system_prompt).add(CHAT_ROLE.USER, user_prompt)#.add(CHAT_ROLE.SYSTEM, system_prompt).add(CHAT_ROLE.USER, user_prompt)
    chat_id = barebonesllmchat_CLIENT.send_history(None, history, blocking=True,
                                                   generation_settings={"max_new_tokens": max_tokens}
                                                   )
    response = barebonesllmchat_CLIENT.get_chat_messages(chat_id)[-1]["content"]

    # attempt to rescue response, sometimes OLMO forgets to say "NAME Alice"
    #for name in ["Alice", "Bob", "Chad", "David"]:
    #    if name in response and f"NAME {name}" not in response and name in user_prompt:
    #        response = response.replace("EXECUTE\n", "EXECUTE\nNAME ")

    assert response is not None
    return response, None


def llm_autocomplete(max_retries, model, system_prompt, user_prompt, max_tokens, temperature):
    _query_func = _query_chatgpt if False else _query_barebonesllmchat

    usage, response = None, None

    for n in range(max_retries):
        print('querying {}th time'.format(n))
        try:
            response, usage = _query_func(model, system_prompt, user_prompt, max_tokens, temperature)
            print('======= response ======= \n ', response)
            print('======= usage ======= \n ', usage)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("API error, try again")
    return response, usage


def _compose_system_prompt(
    self,
    obs: EnvState,
    agent_name: str,
    chat_history: List = [], # chat from previous replan rounds
    current_chat: List = [],  # chat from current round, this comes AFTER env feedback
    feedback_history: List = []
) -> str:
    action_desp = self.env.get_action_prompt()
    if self.use_waypoints:
        action_desp += PATH_PLAN_INSTRUCTION
    agent_prompt = self.env.get_agent_prompt(obs, agent_name)

    round_history = self.get_round_history() if self.use_history else ""

    execute_feedback = ""
    if len(self.failed_plans) > 0:
        execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
        execute_feedback += "\n".join(self.failed_plans) + "\n"

    chat_history = "[Previous Chat]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""

    game_description = """
    You are a player in a robot collaboration game. There are two players in this game. You are one of the players.
    
    [RULES]
    1) The two players alternate.
    2) The two players must select one ACTION each.
    3) A player can only propose actions for themself, but can provide feedback for the other player.
    4) The players must collaborate to select ACTIONS to accomplish a task together.
    5) Each turn, the player must pick one of the PATTERNs described below. Each PATTERN allows them to accomplish different things in the game. The PATTERNS help the players refine their ACTIONS.

    [PATTERNS]
    We will now describe the possible PATTERNS that players might use during their turn.
    (A) PROPOSE: The player first THINKS, writing their thoughts down. Then, the player PROPOSES an ACTION (described below) to add to the plan.
    (B) FEEDBACK AND PROPOSE: The player first THINKS. Then, the player writes down FEEDBACK for the other player. Finally, the player PROPOSES an ACTION (described below).
    (C) APPROVE: Once both players have PROPOSED ACTIONS, they can APPROVE them instead of FEEDBACK AND PROPOSE them.
    (D) APPROVE AND EXECUTE: Once another player has APPROVED the plan, the current player can unilaterally opt to EXECUTE the ACTIONS. The player first writes EXECUTE, then writes down the two proposed ACTIONS.
    When a player selects a PATTERN, they must explicitly say so.
    
    Example:
    ALICE: (A): I can see a bottle in front of me. I plan to pick it. NAME Alice ACTION PICK bottle PATH [(0.29, 0.06, 0.51), (0.29, 0.06, 0.6), (0.45, 0.58, 0.6), (0.45, 0.58, 0.36)]
    BOB: (A): Hey Alice, sounds good! I'll pick the rock since it avoids collisions with you. NAME Bob ACTION PICK rock PATH [(0.35, 1.05, 0.62), (0.35, 0.5, 0.62), (-0.50, 0.59, 0.6), (-0.50, 0.59, 0.49)]
    ALICE: (B): Hey Bob, thanks! But your path seems to intersect with the apple. Maybe raise a bit? NAME Alice ACTION PICK bottle PATH [(0.29, 0.06, 0.51), (0.29, 0.06, 0.6), (0.45, 0.58, 0.6), (0.45, 0.58, 0.36)]
    BOB: (A): You're right, my gripper will be a bit too low! NAME Bob ACTION PICK rock PATH [(0.29, 0.06, 0.55), (0.29, 0.06, 0.65), (0.45, 0.58, 0.65), (0.45, 0.58, 0.36)]
    ALICE: (C): Cool, I APPROVE
    BOB: (D) EXECUTE
    NAME Alice ACTION PICK bottle PATH [(0.29, 0.06, 0.51), (0.29, 0.06, 0.6), (0.45, 0.58, 0.6), (0.45, 0.58, 0.36)]
    NAME Bob ACTION PICK rock PATH [(0.35, 1.05, 0.62), (0.35, 0.5, 0.62), (-0.50, 0.59, 0.6), (-0.50, 0.59, 0.49)]
    
    [ACTIONS]
    1) PICK <obj> PATH <path>: only PICK if your gripper is empty;
    2) PLACE <obj> bin PATH <path>: only if you have already PICKed the object, you can PLACE it into an empty bin slot, do NOT PLACE if another object is already in a slot!
    Each <path> must contain exactly four <coord>s that smoothly interpolate between start and goal, coordinates must be evenly distanced from each other.
    The robot PATHs must efficiently reach target while avoiding collision avoid collision (e.g. move above the objects' heights).
    
    [PATHS]
    Each coordinate in a PATH is a tuple (x,y,z) for gripper location.
    1) Decide target location (e.g. an object you want to pick), and your current gripper location.
    2) Plan a list of <coord> that move smoothly from current gripper to the target location.
    3) The <coord>s must be evenly spaced between start and target.
    4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
    [How to Incoporate [Environment Feedback] to improve plan]
        If IK fails, propose more feasible step for the gripper to reach. 
        If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
        If collision is detected at a Goal Step, choose a different action.
        To make a path more evenly spaced, make distance between pair-wise steps similar.
            e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
        If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
    """

    agent_prompt = agent_prompt.split("\nWhen you")[0]
    system_prompt = f"{game_description}\n\n{agent_prompt}\n\n{round_history}\n{execute_feedback}\n{chat_history}"

    #system_prompt = f"\n{action_desp}\n{round_history}\n{execute_feedback}{agent_prompt}\n{chat_history}\n"

    if self.use_feedback and len(feedback_history) > 0:
        system_prompt += "\n".join(feedback_history)

    if len(current_chat) > 0:
        system_prompt += "[Current Chat]\n" + "\n".join(current_chat) + "\n"

    return system_prompt

##response, usage = llm_autocomplete(max_retries=max_query, model=self.llm_source, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        #return response, usage