import os
import argparse
from PIL import Image
import openai

from tools.utils import prompts
from tools.GPTReasoning import GPT4Reasoning
from tools.matching import TargetMatching, ImageCropsBaseAttributesMatching
from tools.detection import GroundedDetection
from tools.segmentation import DetPromptedSegmentation


class Engine():
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = GroundedDetection(cfg)
        self.segmenter = DetPromptedSegmentation(cfg)
        self.base_matching = ImageCropsBaseAttributesMatching(cfg)


    @prompts(name="extract object crops and corresponding base attributes from image",
            description="useful when you try to understand the image content in a structured way")
    def infer_img_grounded_objects_base_attributes(self, image_path, unique_nouns="stool"):
        image = Image.open(image_path).convert("RGB") 
        boxes, pred_phrases = self.detector.inference(image_path, unique_nouns, 
                                                      self.cfg.box_threshold, self.cfg.text_threshold, self.cfg.iou_threshold)
        masks = self.segmenter.inference(image_path, prompt_boxes=boxes, save_json=False)
        self.base_matching.clip_text_base_attribute_embedding(self.cfg.output_dir)
        objects_base_attributes = self.base_matching.get_objects_base_attributes(image, boxes, pred_phrases, masks) 
        return objects_base_attributes


    @prompts(name="extract target-related information from request",
            description="useful when you try to understand the human request in a structured way")
    def infer_txt_subgraph(self, request):
        #   ----------------- subgraph_base_attributes -----------------
        #   example for single target test: "give me the yellow stool"
        #   subgraph_base_attributes = {'target': [{'name': 'stool', 'color': 'yellow'}], 'related': []}  
        #  
        #   example for s single target, related with other objects.
        #   subgraph_base_attributes = { 'target':  [{'name': 'stool', 'color': 'red'}], 
        #                              'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
        subgraph_dict = GPT4Reasoning(temperature=0).extract_target_relations(request)
        return subgraph_dict



def chatref_main(cfg):
    engine = Engine(cfg=cfg)
    # input inference
    unique_nouns = GPT4Reasoning(temperature=0).extract_unique_nouns(cfg.request)
    # unique_nouns = "stool"
    crops_base_list = engine.infer_img_grounded_objects_base_attributes(cfg.input_image, unique_nouns)
    text_subgraph = engine.infer_txt_subgraph(cfg.request)
    # text_subgraph = {'target': [{'name': 'stool', 'color': 'red'}], 
    #                  'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}

    # matching process
    chatref = TargetMatching(cfg.image_path, cfg.request, crops_base_list, text_subgraph, cfg)
    targets = chatref.inference()
    return



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
    parser.add_argument("--ui", default=False, help="run on gradio UI")
    cfg = parser.parse_args()

    cfg.output_dir = os.path.join(cfg.output_dir, cfg.request)
    os.makedirs(cfg.output_dir, exist_ok=True)

    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file
    cfg.openai_api_key = os.environ['OPENAI_API_KEY']
    openai.api_key = cfg.openai_api_key
    openai_proxy = None
    if openai_proxy:
        openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    chatref_main(cfg)
