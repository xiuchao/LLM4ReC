import os
import re
import argparse
import numpy as np
from PIL import Image
import gradio as gr
import openai
from llm_grasp import Engine, TargetMatching
from tools.utils import read_image_pil
from tools.prompt import gen_extract_target_prompt, gen_extract_target_relation_prompt, \
                        gen_extract_disambiguated_target_prompt
from tools.plot_utils import draw_candidate_boxes
from tools.GPTReferring import GPTReferring

# gradio bot
class ConversationBot:
    def __init__(self, cfg):
        self.cfg = cfg
        print(f"Initializing ChatRef")
        self.llm = GPTReferring()
        self.engine = Engine(cfg=cfg)


    def __call__(self, chat_hist, image_input):
        rawimage_pil = read_image_pil(image_input)
        request = chat_hist[-1][0]
        extract_target_prompt = gen_extract_target_prompt(request)
        target_nouns = self.llm.get_completion(extract_target_prompt)
        target_nouns = eval(target_nouns)
        if target_nouns["has_target"] == False:
            chat_hist[-1][1] = target_nouns["ask_user"]
            return chat_hist, None
        unique_nouns = target_nouns["target"][0]
        # unique_nouns = 'stool'
        crops_base_list = self.engine.infer_img_grounded_objects_base_attributes(image_input, unique_nouns)
        extract_target_graph_prompt = gen_extract_target_relation_prompt(request)
        text_subgraph = self.llm.get_completion(extract_target_graph_prompt)
        text_subgraph = eval(text_subgraph)
        # text_subgraph = {'target': [{'name': 'stool', 'color': 'red'}], 
        #         'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
        if text_subgraph["target"] == []:
            chat_hist[-1][1] = target_nouns["ask_user"]
            return chat_hist, None
        # matching process
        chatref = TargetMatching(image_input, request, crops_base_list, text_subgraph, cfg)
        subgraph_target_list = chatref.match_subgraph_target_base()
        if chatref.count > 1:
            subgraph_related_list = chatref.match_subgraph_related_base(subgraph_target_list)
            if len(subgraph_related_list) > 1:
                extract_disambiguated_target_prompt = gen_extract_disambiguated_target_prompt(request)
                gpt_ref = self.llm.get_completion(extract_disambiguated_target_prompt)
                ref_ids = list(map(int, gpt_ref.split(',')))
                gpt_ref_list = [subgraph_related_list[i] for i in ref_ids]
                print(f'{len(gpt_ref_list)} targets found!')
                display_image = draw_candidate_boxes(rawimage_pil, gpt_ref_list, self.cfg.output_dir, stepstr='gptref', save=True)
                return chat_hist, display_image
            elif len(subgraph_related_list) == 1:
                display_image = draw_candidate_boxes(rawimage_pil, subgraph_related_list, self.cfg.output_dir, stepstr='gptref', save=True)
                return chat_hist, display_image
            else:
                return chat_hist, None
        elif chatref.count == 1:
            print('target object found!')
            display_image = draw_candidate_boxes(rawimage_pil, subgraph_target_list, self.cfg.output_dir, stepstr='gptref', save=True)
            return chat_hist, display_image
        else:
            print('no target object found!')
            return chat_hist, None


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
        clear.click(lambda: [], None, output_image)

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
    openai.api_key = os.environ['OPENAI_API_KEY']

    launch_chat_bot(cfg)
