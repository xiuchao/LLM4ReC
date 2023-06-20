import os
import re
import argparse
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
# uuid
import uuid
import gradio as gr

# langchain
from langchain.memory import ConversationTokenBufferMemory, ConversationBufferWindowMemory
from langchain.llms.openai import OpenAI


from LLM_ReC_yangh import img_inference_grounded_objects_base_attributes, \
    txt_inference_subgraph, TargetMatching


# gradio bot
class ConversationBot:
    def __init__(self, cfg):
        self.cfg = cfg
        print(f"Initializing ChatRef")
        self.llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=cfg.openai_api_key, temperature=0)
        # self.memory = ConversationTokenBufferMemory(max_token_limit=200,
        #                                             memory_key="chat_history", 
        #                                             output_key='output')
        self.memory = ConversationBufferWindowMemory(k=5, 
                                                     memory_key="chat_history", 
                                                     output_key='output')


    def __call__(self, chat_hist, file):
        image_path = file.name
        request = chat_hist[-1][0]
        # crops_base_list = img_inference_grounded_objects_base_attributes(image_path, request, self.cfg)
        # # text_subgraph = txt_inference_subgraph(request)
        # text_subgraph = {'target': [{'name': 'stool', 'color': 'red'}], 
        #         'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
        # # matching process
        # chatref = TargetMatching(image_path, request, crops_base_list, text_subgraph, self.cfg)
        # targets = chatref.inference()
        display_image = os.path.join(self.cfg.output_dir, request, "gptref.jpg")
        if os.path.exists(display_image):
            follow_up = "Is it the correct object you want?"
            chat_hist[-1][1] = follow_up
            return chat_hist, display_image
        return chat_hist, image_path


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(input_image, file):
    # input_image = file.name
    # '/tmp/gradio/345a7b57bee462c4a513c38a8ef0008c5e80cd2a/Ego4d-EpisodicMemory.PNG'
    os.makedirs('image', exist_ok=True) 
    image_filename = os.path.join('image', file.name.split('/')[-1])

    print("======>Auto Resize Image...")
    img = Image.open(file.name)
    width, height = img.size
    ratio = min(512 / width, 512 / height)
    width_new, height_new = (round(width * ratio), round(height * ratio))
    width_new = int(np.round(width_new / 64.0)) * 64
    height_new = int(np.round(height_new / 64.0)) * 64
    img = img.resize((width_new, height_new))
    img = img.convert('RGB')
    img.save(image_filename, "PNG")
    print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
    input_image = img
    return input_image
    

# def bot(history):
#     response = "**That's cool!**"
#     history[-1][1] = response
#     return history

def launch_chat_bot(cfg):
    bot = ConversationBot(cfg)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=0.5):
                input_image = gr.inputs.Image(shape=(256,256)).style(height=400)
            with gr.Column(scale=0.5):
                chat_hist = gr.Chatbot([], elem_id="chatbot", label="ChatRef").style(height=400)
        with gr.Row():
            with gr.Column(scale=0.75):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                img_btn = gr.UploadButton("📁", file_types=["image"])
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
                # clear = gr.ClearButton([btn, chat_hist])

        txt_msg = txt.submit(add_text, [chat_hist, txt], [chat_hist, txt], queue=False).then(
            bot, [chat_hist, img_btn], [chat_hist, input_image]
        )
        txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
        file_msg = img_btn.upload(add_file, [input_image, img_btn], [input_image], queue=False)

        clear.click(lambda: [], None, chat_hist)
        clear.click(lambda: [], None, img_btn)
        clear.click(lambda: [], None, input_image)

        demo.launch(server_name="0.0.0.0", server_port=7992)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser("ChatRef", add_help=True)
    # parser.add_argument("--openai_api_key", type=str, required=False, help="openai key")
    parser.add_argument("--input_image", type=str, default="image/e12572de.png", help="path to image file")
    parser.add_argument("--request", type=str, default="red stool on the left of the yellow stool", help="human request")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--visualize", default=True, help="visualize intermediate data mode")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--ui", default=True, help="run on gradio UI")
    cfg = parser.parse_args()

    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file
    cfg.openai_api_key = os.environ['OPENAI_API_KEY']

    launch_chat_bot(cfg)
