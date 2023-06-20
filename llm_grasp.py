import os
import argparse
from PIL import Image
import openai

from tools.utils import prompts
from tools.GPTReasoning import GPT4Reasoning
from tools.matching import TargetMatching, ImageCropsBaseAttributesMatching
from tools.detection import GroundedDetection
from tools.segmentation import DetPromptedSegmentation



@prompts(name="extract object crops and corresponding base attributes from image",
         description="useful when you try to understand the image content in a structured way")
def img_inference_grounded_objects_base_attributes(image_path, request, cfg):
    image_pil = Image.open(image_path).convert("RGB") 
    unique_nouns = GPT4Reasoning(temperature=0).extract_unique_nouns(request)
    # unique_nouns = "stool"

    detector = GroundedDetection(cfg)
    boxes, pred_phrases = detector.inference(image_path, unique_nouns, cfg.box_threshold, cfg.text_threshold, cfg.iou_threshold)
    segmenter = DetPromptedSegmentation(cfg)
    masks = segmenter.inference(image_path, prompt_boxes=boxes, save_json=False)

    matcher_base = ImageCropsBaseAttributesMatching(cfg)
    objects_base_attributes = matcher_base.get_objects_base_attributes(image_pil, boxes, pred_phrases, masks) 
    return objects_base_attributes

@prompts(name="extract target-related information from request",
         description="useful when you try to understand the human request in a structured way")
def txt_inference_subgraph(request):
    #   ----------------- subgraph_base_attributes -----------------
    #   example for single target test: "give me the yellow stool"
    #   subgraph_base_attributes = {'target': [{'name': 'stool', 'color': 'yellow'}], 'related': []}  
    #  
    #   example for s single target, related with other objects.
    #   subgraph_base_attributes = { 'target':  [{'name': 'stool', 'color': 'red'}], 
    #                              'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
    subgraph_dict = GPT4Reasoning(temperature=0).extract_target_relations(request)
    return subgraph_dict


def chatref_main(image_path, request, cfg):
    # input inference
    crops_base_list = img_inference_grounded_objects_base_attributes(image_path, request, cfg)
    text_subgraph = txt_inference_subgraph(request)
    # text_subgraph = {'target': [{'name': 'stool', 'color': 'red'}], 
    #                  'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}

    # matching process
    chatref = TargetMatching(image_path, request, crops_base_list, text_subgraph, cfg)
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

    image_path = cfg.input_image
    request = cfg.request
    cfg.output_dir = os.path.join(cfg.output_dir, request)
    os.makedirs(cfg.output_dir,exist_ok=True)

    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file
    cfg.openai_api_key = os.environ['OPENAI_API_KEY']
    openai.api_key = cfg.openai_api_key
    openai_proxy = None
    if openai_proxy:
        openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    chatref_main(image_path, request, cfg)
