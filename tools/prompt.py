import json


def remove_whitespace(input_str):
    input_str = input_str.replace("    ", "")
    input_str = input_str.replace("\t", "")
    return input_str


def gen_rephrase_semantic_question_prompt(request, chat_hist):
    '''Args:
    request: str, human's question without role prefix.
    chat_hist: str, human and AI's chat history with role prefix.
    '''
    prompt = f"""
    Given the following conversation and a follow up question, summarize and rephrase human's question \
    to be a standalone question.
    Chat History: {chat_hist}
    Follow Up Input: {request}
    Standalone question:
    """
    prompt = remove_whitespace(prompt)
    return prompt


def gen_extract_target_prompt(request, delimiter="###"):
    extract_target = f"""
    Figure out is there any target object that the human want to find given the following chat history delimited by {delimiter}.
    Chat history: {delimiter}{request}{delimiter}
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
    # Figure out is there any target object that the human want to find given the following chat history delimited by {delimiter}.
    # Chat history: {delimiter}{request}{delimiter}
    # - If no, output a json object using the following format:
    # {{"has_target": False, \
    # "target": [], \
    # "ask_user": <reply to the human's request>}}
    # - If yes, extract the target object, related objects (exclude the target) and spatial relationships from the human request. \
    # output a json object using the following format:
    # {{"has_target": True, \
    # "target": [{{"name":<target in singular form without adjective>, "color":str, "shape":str, "spatial":str}}], \
    # "related": [{{"name": str, "color": str, "spatial": str, target: int}}], \
    # "ask_user": ""}}
    # Note: 'something' should not be considered as a target. Do not print the input. \
    # Here is an example: \
    # Human: give the red cup next to the red apple, the spoon on the left of the square plate.
    # {sample_json}
    # """
    extract_target_relation = f"""{request}
    Figure out is there any target object that the human want to find in the question.
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
    '''Args:
    request: str, human's question without role prefix.
    image_size: str, (height, width)
    crops_attributes_list: str, [{'name': ['stool', 0.57], 'box': [321, 54, 494, 244], 
                                'width': 173, 'height': 190, 'color': ('red', 0.86), 
                                'shape': ('cuboid', 0.37), 'texture': ('metal', 0.34)}, 
                                {'name': ...}]
    '''
    # extract_disambiguated_target = f"""
    # You're chatting with a human who will give you some requests to find target object(s). \
    # Here are the previous chat history: {delimiter}{request}{delimiter}
    # and other information: \
    # Image size: {image_size}. \
    # Candidate objects: {crops_attributes_list}.
    # You must do the following steps:
    # 1 - determine the target object(s) that the human want. \
    # 2 - analyse box locations or other attributes of candidate objects. \
    # 3 - locate the target(s) from 'Condidate objects' list. Return the order index, eg. 0, 1, 2...
    # Output a json object using the format: \
    # {{"result": <the order index, list>, \
    # "explain": explain the reason}}
    # """
    '''{"has_target": True, "target": [{"name": "stool", "color": "red"}], "related": [{"spatial": "on the left of", "target": "0", "name": "stool", "color": "yellow"}], "ask_user": ""}
    '''
    extract_disambiguated_target = f"""{request}
    Figure out the target object(s) that the human want to find with the following information: \
    Image size: {image_size}. \
    Candidate objects: {crops_attributes_list}.
    You must do the following steps:
    1 - determine the target object(s) that the human want. \
    2 - analyse box locations or other attributes of candidate objects. \
    3 - locate the target(s) from 'Condidate objects' list. Return the order index, eg. 0, 1, 2... \
    If there is no target in the candidate objects list, set the output result greater than the length \
    of the candidate list. \
    Output a json object using the format: \
    {{"result": <the order index, list>, \
    "explain": explain the reason}}
    """
    extract_disambiguated_target = remove_whitespace(extract_disambiguated_target)
    return extract_disambiguated_target



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