import json


def remove_whitespace(input_str):
    input_str = input_str.replace("    ", "")
    input_str = input_str.replace("\t", "")
    return input_str


def gen_extract_target_prompt(request, delimiter="###"):
    extract_target = f"""
    Figure out is there any target object that the human want to find given the following chat history delimited by {delimiter}.
    Chat history: {delimiter}{request}{delimiter}.
    - If no, output a json object using the following format:
    {{"has_target": False, \
    "target": "", \
    "ask_user": ask user to specify or describe the target}}
    - If yes, output a json object using the following format:
    {{"has_target": True, \
    "target": <List the target in singular form without adjective>, \
    "ask_user": ""}}
    Note: 'something' should not be considered as a target. Do not print the input.
    """
    '''
    {"has_target": False, "target": "", "ask_user": "Please specify or describe the target you are looking for."}

    {"has_target": True, "target": "stool", "ask_user": ""}
    '''
    extract_target = remove_whitespace(extract_target)
    return extract_target


def gen_extract_target_relation_prompt(request, delimiter="###"):
    sample_json = {"target": [{"name": "cup", "color": "red"}, {"name": "spoon"}],
                    "related": [{"spatial": "next to", "target":"0", "name": "apple", "color": "red"},
                                {"spatial": "on the left of", "target":"1", "name": "plate", "shape": "square"}]
                }
    # extract_target_relation = f"""
    # Extract the target object, related objects and spatial relationships from the human request. \
    # Specify the object in singular mode, e.g. two red apples as (red apple, red apple)
    # Output follow standard JSON format. Here is an example: \
    # Human: give the red cup next to the red apple, the spoon on the left of the square plate.
    # {sample_json}
    # Human: {request}.
    # Note: 'something' should not be considered as a target. \
    # If there's no target found, output a json object using the following format:
    # {{"target": [], "ask_user": ask user to specify or describe the target}}
    # Do not print the input.
    # """
    '''
    {'target': [{'name': 'stool', 'color': 'red'}], 'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
    {"target": [], "ask_user": "Please specify or describe the target object"}
    '''
    # extract_target_relation = f"""
    # Figure out is there any target object that the human want to find given the following chat history delimited by {delimiter}.
    # Chat history: {delimiter}{request}{delimiter}.
    # Extract the target object, related objects (exclude the target) and spatial relationships from the human request. \
    # - If no, output a json object using the following format:
    # {{"has_target": False, \
    # "target": [], \
    # "ask_user": ask user to specify or describe the target}}
    # - If yes, output a json object using the following format:
    # {{"has_target": True, \
    # "target": [{{"name":<target in singular form without adjective>, "color":str, "shape":str, "spatial":str}}], \
    # "related": [{{"name": str, "color": str, "spatial": str, target: int}}], \
    # "ask_user": ""}}
    # Note: 'something' should not be considered as a target. Do not print the input. \
    # Here is an example: \
    # Human: give the red cup next to the red apple, the spoon on the left of the square plate.
    # {sample_json}
    # """
    '''
    red stool on the left of the yellow stool
    {"has_target": True, "target": [{"name": "stool", "color": "red"}], "related": [{"name": "stool", "color": "yellow", "spatial": "on the left of"}], "ask_user": ""}
    give me the red chair
    {"has_target": True, "target": [{"name": "chair", "color": "red"}], "related": [], "ask_user": ""}
    '''
    extract_target_relation = f"""
    Figure out is there any target object that the human want to find given the following chat history delimited by {delimiter}.
    Chat history: {delimiter}{request}{delimiter}.
    - If no, output a json object using the following format:
    {{"has_target": False, \
    "target": [], \
    "ask_user": <reply to the human's request>}}
    - If yes, extract the target object, related objects (exclude the target) and spatial relationships from the human request. \
    output a json object using the following format:
    {{"has_target": True, \
    "target": [{{"name":<target in singular form without adjective>, "color":str, "shape":str, "spatial":str}}], \
    "related": [{{"name": str, "color": str, "spatial": str, target: int}}], \
    "ask_user": ""}}
    Note: 'something' should not be considered as a target. Do not print the input. \
    Here is an example: \
    Human: give the red cup next to the red apple, the spoon on the left of the square plate.
    {sample_json}
    """
    extract_target_relation = remove_whitespace(extract_target_relation)
    return extract_target_relation


def gen_extract_disambiguated_target_prompt(request, image_size, crops_attributes_list, delimiter="###"):
    # extract_disambiguated_target = f"""
    # Locate the target object(s) according to the chat history and the following information.
    # Chat history: {request}.
    # Image size: {image_size}. \
    # Candidate objects: {crops_attributes_list}.
    # You must do the following steps:
    # 1 - determine the target object(s) that the user want based on the 'Chat history'. \
    # 2 - locate the target(s) from 'Condidate objects' list.
    # Output only the order index, like 0, 1, 2...
    # """
    '''
    Based on the chat history, the user is looking for the red stool on the left of the yellow stool. 

To locate the target object(s) from the candidate objects list, we need to find the red stool that is positioned to the left of the yellow stool.

Looking at the candidate objects list, we can see that there are two red stools with different positions. The first red stool has a box coordinates of [321, 54, 494, 244] and the second red stool has a box coordinates of [138, 139, 237, 242].

Comparing the positions of the red stools with the yellow stool, we can determine that the second red stool with index 1 is the one on the left of the yellow stool.

Therefore, the target object is the red stool with index 1.
    '''

    # extract_disambiguated_target = f"""
    # Locate the target object(s) according to the chat history and the following information.
    # Chat history: {request}.
    # Image size: {image_size}. \
    # Candidate objects: {crops_attributes_list}.
    # You must do the following steps:
    # 1 - determine the target object(s) that the user want based on the 'Chat history'. \
    # 2 - locate the target(s) from 'Condidate objects' list. Return the order index, like 0, 1, 2...
    # Output a json object using the format: \
    # {{"result": <the order index, int>, \
    # "explain": explain in details}}
    # """
    '''{"result": 1, "explain": "The target object is the red stool on the left of the yellow stool. Based on the chat history, the red stool is mentioned first and it is specified to be on the left of the yellow stool. Looking at the candidate objects, the second object in the list matches this description. Therefore, the order index of the target object is 1."}
    '''
    # extract_disambiguated_target = f"""
    # Locate the target object(s) according to the chat history and the following information.
    # Chat history: {request}.
    # Image size: {image_size}. \
    # Candidate objects: {crops_attributes_list}.
    # You must do the following steps:
    # 1 - determine the target object(s) that the user want based on the 'Chat history'. \
    # 2 - locate the target(s) from 'Condidate objects' list. Return the order index, like 0, 1, 2...
    # Output a json object using the format: \
    # {{"result": <the order index, list>, \
    # "explain": explain the reason}}
    # """
    '''{"result": 1, "explain": "Based on the chat history, the target object is the red stool on the left of the yellow stool. Looking at the candidate objects, the second object in the list matches this description. Therefore, the order index is 1."}'''

    # extract_disambiguated_target = f"""
    # Locate the target object(s) according to the chat history between a human and AI (you) \
    # and the following information.
    # Chat history: {delimiter}{request}{delimiter}
    # Image size: {image_size}. \
    # Candidate objects: {crops_attributes_list}.
    # You must do the following steps:
    # 1 - determine the target object(s) that the human want based on the 'Chat history'. \
    # 2 - match objects with candidate objects with their properties and relations. \
    # 3 - locate the target(s) from 'Condidate objects' list. Return the order index, eg. 0, 1, 2...
    # Output a json object using the format: \
    # {{"result": <the order index, list>, \
    # "explain": explain the reason}}
    # """
    '''{"result": [0], "explain": "The target object is the red stool on the left of the yellow stool. Based on the chat history, the human mentioned a 'red stool' and specified its position as 'on the left of the yellow stool'. Among the candidate objects, there is only one red stool (object 0) and it is positioned on the left of the yellow stool (object 2). Therefore, the target object is object 0."}'''
    
    extract_disambiguated_target = f"""
    You're chatting with a human who will give you some requests to find target object(s). \
    Here are the previous chat history: {delimiter}{request}{delimiter}
    and other information: \
    Image size: {image_size}. \
    Candidate objects: {crops_attributes_list}.
    You must do the following steps:
    1 - determine the target object(s) that the human want. \
    2 - analyse box locations or other attributes of candidate objects. \
    3 - locate the target(s) from 'Condidate objects' list. Return the order index, eg. 0, 1, 2...
    Output a json object using the format: \
    {{"result": <the order index, list>, \
    "explain": explain the reason}}
    """
    '''{"has_target": True, "target": [{"name": "stool", "color": "red"}], "related": [{"spatial": "on the left of", "target": "0", "name": "stool", "color": "yellow"}], "ask_user": ""}
    '''
    extract_disambiguated_target = remove_whitespace(extract_disambiguated_target)
    return extract_disambiguated_target


def gen_rephrase_semantic_question_prompt(request, chat_hist):
    prompt = f"""
    Given the following conversation and a follow up question, rephrase the follow up question \
    to be a standalone question.
    Chat History: {chat_hist}
    Follow Up Input: {request}
    Standalone question:
    """
    prompt = remove_whitespace(prompt)
    return prompt


# prompt = {
#     "extract_target": extract_target_4,
#     "extract_target_relation": extract_target_relation,
#     "extract_disambiguated_target": extract_disambiguated_target
# }


# # request="Human: red stool on the left of the yellow stool\n"
# # request="Human: give me the red chair\n"
# request="Human: output a question for effective disambiguation"
# image_size="(512, 256)"
# crops_attributes_list="[{'name': ['stool', 0.57], 'box': [321, 54, 494, 244], 'width': 173, 'height': 190, 'color': ('red', 0.86), 'shape': ('cuboid', 0.37), 'texture': ('metal', 0.34)}, {'name': ['stool', 0.51], 'box': [138, 139, 237, 242], 'width': 99, 'height': 103, 'color': ('red', 0.61), 'shape': ('cuboid', 0.49), 'texture': ('wood', 0.27)}, {'name': ['stool', 0.57], 'box': [220, 119, 348, 244], 'width': 128, 'height': 125, 'color': ('yellow', 0.84), 'shape': ('cuboid', 0.38), 'texture': ('wood', 0.39)}]"

# # extract_disambiguated_target=gen_extract_disambiguated_target_prompt(request, image_size, crops_attributes_list)
# # print(extract_disambiguated_target)

# extract_target_graph_prompt = gen_extract_target_relation_prompt(request)
# print(extract_target_graph_prompt)