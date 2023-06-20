import json
import openai


class GPT4Reasoning: 
    def __init__(self, split=',', delimiter="###", max_tokens=100, temperature=0.6):
        self.llms = ["gpt-3.5-turbo", "gpt-4"]
        self.split = split
        self.delimiter = delimiter
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt = [{'role': 'user', 'content': ''}]


    def extract_unique_nouns(self, request): 
        self.prompt[0]['content'] = f"""
        Extract the unique objects provided by the following caption delimited by {self.delimiter}. 
        Remove all adjectives.
        List the objects in singular form and delimite them by comma.
        Caption: {self.delimiter}{request}{self.delimiter}.
        """
        
        response = openai.ChatCompletion.create(model=self.llms[0], 
                                                messages=self.prompt, 
                                                temperature=self.temperature, 
                                                max_tokens=self.max_tokens)
        reply = response['choices'][0]['message']['content']
        unique_nouns = reply.split(':')[-1].strip() # sometimes return with "noun: xxx, xxx, xxx"
        print("BOT:", unique_nouns)
        return unique_nouns

    # reasoning related objects: subgraph_dict
    def extract_target_relations(self, request): 
        # Request: "give the red apple between the blue cup and round white plate."
        # Request: "give the red apple and yellow apple between the blue cup and the round white plate.
        sample_json = {"target": [{"name": "cup", "color": "red"}, {"name": "spoon"}],
                        "related": [{"spatial": "next to", "target":"0", "name": "apple", "color": "red"},
                                    {"spatial": "on the left of", "target":"1", "name": "plate", "shape": "square"}]
                    }
        self.prompt[0]['content'] = f"""
        Extract the target object, related objects and spatial relationships from the human request.
        Specify the object in singular mode, e.g. two red apples as (red apple, red apple)
        Output follow standard JSON format. Here is an example:
        request: give the red cup next to the red apple, the spoon on the left of the square plate.
        {sample_json}
        Human request: {request}.
        """

        response = openai.ChatCompletion.create(model=self.llms[0], 
                                                messages=self.prompt, 
                                                temperature=self.temperature, 
                                                max_tokens=self.max_tokens)
        reply = response['choices'][0]['message']['content']
        def json2dict(input_str):
            input_str = input_str.replace("'", '"').strip()
            data = json.loads(input_str)
            return data
        subgraph_dict = json2dict(reply)
        return subgraph_dict

    # locate target, disambiguation
    def extract_disambiguated_target(self, request, crops_attributes_list, image_size=(None, None)):
        # self.prompt[0]['content'] = f"""
        # Locate the target object(s) from human request according to the given information.
        # The image size is {image_size}.
        # Human request: {request}.
        # Candidate objects: {crops_attributes_list}.
        # Output only the index number.
        # """
        self.prompt[0]['content'] = f"""
        Locate the target object(s) in the request according to the given information.'
        The image size is {image_size}. Output only the order index, like 0, 1, 2...'
        Human request: {request}.
        Candidate objects: {crops_attributes_list}.
        """
        response = openai.ChatCompletion.create(model=self.llms[0], 
                                                messages=self.prompt, 
                                                temperature=self.temperature, 
                                                max_tokens=500)
        reply = response['choices'][0]['message']['content']
        return reply

    def get_counted_BLIP(self, caption, pred_phrases):
        object_list = [obj.split('(')[0] for obj in pred_phrases]
        object_num = []
        for obj in set(object_list):
            object_num.append(f'{object_list.count(obj)} {obj}')
        object_num = ', '.join(object_num)
        print(f"Correct object number: {object_num}")

        self.prompt[0]['content'] = f"""
        Revise the number in the caption if it is wrong.
        Caption: {caption}.
        True object number: {object_num}.
        Only give the revised caption.
        """
        
        response = openai.ChatCompletion.create(model=self.llms[0], 
                                                messages=self.prompt, 
                                                temperature=self.temperature, 
                                                max_tokens=self.max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "Caption: xxx, xxx, xxx"
        caption = reply.split(':')[-1].strip()
        return caption
