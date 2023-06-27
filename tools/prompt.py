import json


delimiter = "###"
request = "red stool on the left of the yellow stool"
# request = "give me something"

extract_target_1 = f"""
You are a chatbot to try to help human to find the target object in an image. Descriptions of the target \
given by human sometimes can be ambiguous. \
Perform the following actions:
1 - Determine whether the user's question is about the designated target. \
If no, ask user to specify the target in the upload image. \
If yes, extract the unique objects provided by the following caption delimited by {delimiter}. \
Remove all adjectives. List the objects in singular form and delimite them by comma.
Caption: {delimiter}{request}{delimiter}.
"""

extract_target_2 = f"""
Figure out is there any target object that the human want to find given the following caption delimited by {delimiter}.
Caption: {delimiter}{request}{delimiter}.
Perform the following actions:
Determine whether the user's question is about the designated target. \
If no, ask user to specify the target in the upload image and return 'has_target' False. \
If yes, extract the unique objects and remove all adjectives. \
List the objects in singular form and delimite them by comma. Return 'has_target' True.
Output a json object using the following format:
{{"has_target": <True or False>, \
"target": <objects string>, \
"ask_user": <string>}}
"""
"""
{"has_target": True, "target": "stool, stool", "ask_user": ""}
a = '{"has_target": True, "target": "stool, stool", "ask_user": ""}'
b = eval(a)
"""

extract_target_3 = f"""
Figure out is there any target object that the human want to find given the following caption delimited by {delimiter}.
Caption: {delimiter}{request}{delimiter}.
Perform the following actions:
Determine whether the user's caption is about the designated target. \
If no, ask user to specify the target and return 'has_target' False. \
If yes, extract the unique target and remove all adjectives. \
List the target in singular form and delimite them by comma. Return 'has_target' True.
Output a json object using the following format:
{{"has_target": <True or False>, \
"target": <objects string>, \
"ask_user": <string>}}
Do not print the input.
"""
'''
Input: "give me something"

Output: {"has_target": False, "target": "", "ask_user": "What is the target object you are looking for?"}
'''

def remove_whitespace(input_str):
    input_str = input_str.replace("    ", "")
    input_str = input_str.replace("\t", "")
    return input_str


def gen_extract_target_prompt(request, delimiter="###"):
    extract_target = f"""
    Figure out is there any target object that the human want to find given the following caption delimited by {delimiter}.
    Caption: {delimiter}{request}{delimiter}.
    - If no, output a json object using the following format:
    {{"has_target": False, \
    "target": "", \
    "ask_user": ask user to specify or describe the target}}
    - If yes, output a json object using the following format:
    {{"has_target": True, \
    "target": <List the target in singular form and delimite them by comma>, \
    "ask_user": ""}}
    Note: 'something' should not be considered as a target. Do not print the input.
    """
    '''
    {"has_target": False, "target": "", "ask_user": "Please specify or describe the target you are looking for."}

    {"has_target": True, "target": ["stool"], "ask_user": ""}
    '''
    extract_target = remove_whitespace(extract_target)
    return extract_target


def gen_extract_target_relation_prompt(request):
    sample_json = {"target": [{"name": "cup", "color": "red"}, {"name": "spoon"}],
                    "related": [{"spatial": "next to", "target":"0", "name": "apple", "color": "red"},
                                {"spatial": "on the left of", "target":"1", "name": "plate", "shape": "square"}]
                }
    extract_target_relation = f"""
    Extract the target object, related objects and spatial relationships from the human request. \
    Specify the object in singular mode, e.g. two red apples as (red apple, red apple)
    Output follow standard JSON format. Here is an example: \
    Human: give the red cup next to the red apple, the spoon on the left of the square plate.
    {sample_json}
    Human: {request}.
    Note: 'something' should not be considered as a target. \
    If there's no target found, output a json object using the following format:
    {{"target": [], "ask_user": ask user to specify or describe the target}}
    Do not print the input.
    """
    '''
    {'target': [{'name': 'stool', 'color': 'red'}], 'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
    {"target": [], "ask_user": "Please specify or describe the target object"}
    '''
    extract_target_relation = remove_whitespace(extract_target_relation)
    return extract_target_relation


def gen_extract_disambiguated_target_prompt(request, image_size, crops_attributes_list):
    extract_disambiguated_target = f"""
    Locate the target object(s) in the request according to the given information.'
    The image size is {image_size}. Output only the order index, like 0, 1, 2...'
    Human request: {request}.
    Candidate objects: {crops_attributes_list}.
    """
    extract_disambiguated_target = remove_whitespace(extract_disambiguated_target)
    return extract_disambiguated_target



# prompt = {
#     "extract_target": extract_target_4,
#     "extract_target_relation": extract_target_relation,
#     "extract_disambiguated_target": extract_disambiguated_target
# }


