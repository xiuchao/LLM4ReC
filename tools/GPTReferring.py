import json
import openai


class GPTReferring: 
    def __init__(self, window_memory_size=10, delimiter="###", logger=None):
        self.llms = ["gpt-3.5-turbo", "gpt-4"]
        self.delimiter = delimiter
        self.message = [{'role': 'user', 'content': ''}]
        self.memory = []
        self.window_memory_size = window_memory_size
        self.chat_hist = []
        self.logger = logger


    def get_completion(self, prompt, model="gpt-3.5-turbo", 
                       max_tokens=200, temperature=0, 
                       use_memory=True):
        messages = [{"role": "user", "content": prompt}]
        self.update_window_memory(messages)
        if use_memory:
            input_prompt = self.memory
        else:
            input_prompt = messages
        response = openai.ChatCompletion.create(
            model=model,
            messages=input_prompt,
            max_tokens=max_tokens,
            temperature=temperature # this is the degree of randomness of the model's output
        )
        ai_message = [{"role": "assistant", "content": response.choices[0].message["content"]}]
        # ai_chat = "AI: " + response.choices[0].message["content"]
        self.update_window_memory(ai_message)
        # logging
        self.logger.info("prompt:\n{}".format(prompt))
        self.logger.info("response:\n{}".format(response.choices[0].message["content"]))
        return response.choices[0].message["content"]
    

    def get_completion_messages(self, messages, model="gpt-3.5-turbo", 
                                max_tokens=200, temperature=0,
                                use_memory=True):
        self.update_window_memory(messages)
        if use_memory:
            input_prompt = self.memory
        else:
            input_prompt = messages
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=input_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        ai_message = [{"role": "assistant", "content": response.choices[0].message["content"]}]
        self.update_window_memory(ai_message)
        # logging
        self.logger.info("prompt:\n{}".format(messages["content"]))
        self.logger.info("response:\n{}".format(response.choices[0].message["content"]))
        return response.choices[0].message["content"]


    def update_window_memory(self, message: list):
        if len(self.memory) >= self.window_memory_size:
            self.memory.pop(0)
        self.memory.extend(message)
        return

