import json
import openai


class GPTReferring: 
    def __init__(self, window_memory_size=10, delimiter="###"):
        self.llms = ["gpt-3.5-turbo", "gpt-4"]
        self.delimiter = delimiter
        self.message = [{'role': 'user', 'content': ''}]
        self.memory = []
        self.window_memory_size = window_memory_size


    def get_completion(self, prompt, model="gpt-3.5-turbo", max_token=200, temperature=0):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_token=max_token,
            temperature=temperature # this is the degree of randomness of the model's output
        )
        self.update_window_memory(messages)
        self.update_window_memory(response.choices[0].message)
        return response.choices[0].message["content"]
    

    def get_completion_messages(self, messages, model="gpt-3.5-turbo", max_token=200, temperature=0):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_token=max_token,
            temperature=temperature,
        )
        self.update_window_memory(messages)
        self.update_window_memory(response.choices[0].message)
        return response.choices[0].message["content"]


    def update_window_memory(self, message):
        if len(self.memory) >= self.window_memory_size:
            self.memory.pop(0)
        self.memory.append(message)
        return


