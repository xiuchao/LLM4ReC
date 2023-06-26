import os
import re
import argparse
import numpy as np
from PIL import Image
# uuid
import uuid
import gradio as gr
# langchain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.openai import OpenAI

from tools.GPTReasoning import GPT4Reasoning
from llm_grasp import Engine, TargetMatching
from tools.utils import read_image_pil


# gradio bot
class ConversationBot:
    def __init__(self, cfg):
        self.cfg = cfg
        print(f"Initializing ChatRef")
        self.llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=cfg.openai_api_key, temperature=0)
        self.memory = ConversationBufferWindowMemory(k=5, 
                                                     memory_key="chat_history", 
                                                     output_key='output')
        self.engine = Engine(cfg=cfg)


    def __call__(self, chat_hist, image_input):
        request = chat_hist[-1][0]
        self.cfg.request = request
        image_pil = read_image_pil(image_input)
        unique_nouns = GPT4Reasoning(temperature=0).extract_unique_nouns(request)
        # unique_nouns = 'stool'
        crops_base_list = self.engine.infer_img_grounded_objects_base_attributes(image_input, unique_nouns)
        text_subgraph = self.engine.infer_txt_subgraph(request)
        # text_subgraph = {'target': [{'name': 'stool', 'color': 'red'}], 
        #         'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
        # matching process
        chatref = TargetMatching(image_pil, request, crops_base_list, text_subgraph, cfg)
        targets, display_image = chatref.inference()

        if display_image is not None:
            follow_up = "Is it the correct object you want?"
            chat_hist[-1][1] = follow_up
            return chat_hist, display_image
        return chat_hist, image_input


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def process_img(image_pil: Image.Image):
    print("======>Auto Resize Image...")
    width, height = image_pil.size
    ratio = min(512 / width, 512 / height)
    width_new, height_new = (round(width * ratio), round(height * ratio))
    width_new = int(np.round(width_new / 64.0)) * 64
    height_new = int(np.round(height_new / 64.0)) * 64
    image_pil = image_pil.resize((width_new, height_new))
    image_pil = image_pil.convert('RGB')
    print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
    return image_pil
    

# def bot(history):
#     response = "**That's cool!**"
#     history[-1][1] = response
#     return history

def launch_chat_bot(cfg):
    bot = ConversationBot(cfg)
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="numpy")
                output_image = gr.outputs.Image(type="pil", label="results")
            with gr.Column():
                chat_hist = gr.Chatbot([], elem_id="chatbot", label="ChatRef").style(height=500)
                with gr.Row().style(equal_height=True):
                    with gr.Column(scale=0.85):
                        txt = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter",
                        ).style(container=False)
                    # with gr.Column(scale=0.15, min_width=0):
                    #     img_btn = gr.UploadButton("📁", file_types=["image"])
                    with gr.Column(scale=0.15, min_width=0):
                        clear = gr.Button("Clear")

        txt_msg = txt.submit(add_text, [chat_hist, txt], [chat_hist, txt], queue=False).then(
            bot, [chat_hist, input_image], [chat_hist, output_image]
        )
        txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
        clear.click(lambda: [], None, chat_hist)

    block.launch(server_name="0.0.0.0", server_port=7992, debug=True)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser("ChatRef", add_help=True)
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
