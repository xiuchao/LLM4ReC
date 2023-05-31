import argparse
import os
import re
import openai
openai.api_key = "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"

# re, string2dict
def re_response(input_str):
    # input:  "target: {stool: yellow}, related: {stool: red, stool: red}, spatial: {between}"
    # out: {'target': ['stool: yellow'], 'related': ['stool: red', 'stool: red'], 'spatial': ['between']}
    target_match =  re.search(r"(?<=target: \{)(.*)(?=\}, related: \{)", input_str).group()
    related_match = re.search(r"(?<=related: \{)(.*)(?=\}, spatial: \{)", input_str).group()
    spatial_match = re.search(r"(?<=spatial: \{)(.*)(?=\})", input_str).group()
    
    result = {'target':[], 'related':[], 'spatial':[]}
    result['target']  = [item for item in  target_match.split(", ")]
    result['related'] = [item for item in related_match.split(", ")]
    result['spatial'] = [item for item in spatial_match.split(", ")]
    return result


# ChatGPT, reasoning target-related objects.
def GPT_reasoning_objects_relations(caption, split=',', max_tokens=100, model="gpt-4"): 
    prompt = [{ 
        'role': 'system',
        'content': 'Extract the target object, related objects and spatial relationships from the human request.' + \
            'specify the object in singular mode, e.g. two red apples as (red apple, red apple)' + \
            'Output with format as the following:' + \
            'target:{noun:attributes}, related: {noun: attributes}, spatial: {}' + \
            f'Human request: {caption}.'
        }]

    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply_raw = response['choices'][0]['message']['content']
    reply = re_response(reply_raw)
    return reply


if __name__ == "__main__":
    Human = "Give the yellow stool between the two red stools. " 
    objdict = GPT_reasoning_objects_relations(Human)
    caption = GPT_check_caption(caption, pred_phrases)

    reply_raw = 'target: {stool: yellow}, related: {stool: red, stool: red}, spatial: {between}'
    reply = re_response(reply_raw)
