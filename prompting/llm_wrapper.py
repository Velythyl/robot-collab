
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

    PREFIX = """
    You are an agent that is part of a robot task. The two robots must collaborate and communicate with each other.
    Below, you will find a list of rules for the interactions. Please follow those rules very closely.
    You'll notice the OVERSEER can output EXECUTE. Don't forget to make the OVERSEER intervene instead of generating messages as normal.
    Also, you can only generate a message for ONE entity, be it a robot or the OVERSEER.
    """
    #MIDFIX = """\n[Overview]
    #You are an agent that is part of a robot task. The two robots must collaborate and communicate with each other.
    #Below, you will find a list of rules for the interactions. Please follow those rules very closely.
    #You'll notice that EXECUTE indicates a final plan. You can only output EXECUTE if you have preliminary plans for both robots.
    #Also, make sure you PICK an object before PLACEing it.
    #Again, do not output EXECUTE if you do not have plans for both robots. EXECUTE should also include plans for both robots.
    #\n\n"""
    history = ChatHistory().add(CHAT_ROLE.SYSTEM, PREFIX).add( CHAT_ROLE.USER, system_prompt) #.add(CHAT_ROLE.USER, user_prompt)#.add(CHAT_ROLE.SYSTEM, system_prompt).add(CHAT_ROLE.USER, user_prompt)
    chat_id = barebonesllmchat_CLIENT.send_history(None, history, blocking=True,
                                                   generation_settings={"max_new_tokens": max_tokens}
                                                   )
    response = barebonesllmchat_CLIENT.get_chat_messages(chat_id)[-1]["content"]

    #response = response.split("\n\n")[0]    # removes extra responses that the LLM sometimes outputs

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

