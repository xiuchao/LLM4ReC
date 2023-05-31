import argparse
import os
import copy
import json

import torch
import numpy as np
import cv2
import torchvision
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ChatGPT, BLIP
import openai
from transformers import BlipProcessor, BlipForConditionalGeneration

# grdino, SAM
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 

# input/output
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")  # load image
    transform = T.Compose(
        [   T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def save_mask_data(output_dir, caption, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'caption': caption,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)
    
# grounding-dino
def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path) # GroundingDINO Config
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

# BLIP, captioning
def generate_caption(raw_image):
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16) 
    out = blip_model.generate(**inputs) # BLIP model
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ChatGPT, filter text.
def generate_tags(caption, split=',', max_tokens=100, model="gpt-4"): #gpt-3.5-turbo
    # ", " is better for detecting single tags while ". " is a little worse in some case
    prompt = [
        {   'role': 'system',
            'content': 'Extract the unique nouns in the caption. Remove all the adjectives. ' + \
                       f'List the nouns in singular form. Split them by "{split} ". ' + \
                       f'Caption: {caption}.'
        }
    ]
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    tags = reply.split(':')[-1].strip() # sometimes return with "noun: xxx, xxx, xxx"
    return tags

def check_caption(caption, pred_phrases, max_tokens=100, model="gpt-4"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    prompt = [
        {
            'role': 'system',
            'content': 'Revise the number in the caption if it is wrong. ' + \
                       f'Caption: {caption}. ' + \
                       f'True object number: {object_num}. ' + \
                       'Only give the revised caption: '
        }
    ]
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    # sometimes return with "Caption: xxx, xxx, xxx"
    caption = reply.split(':')[-1].strip()
    return caption


if __name__ == "__main__":
    if True:
        # argparse
        parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
        parser.add_argument("--config", type=str, required=True, help="path to config file")
        parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
        parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to checkpoint file")
        parser.add_argument("--input_image", type=str, required=True, help="path to image file")
        parser.add_argument("--split", default=",", type=str, help="split for text prompt")
        parser.add_argument("--openai_key", type=str, required=True, help="key for chatgpt")
        parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
        parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

        parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
        parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
        parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
        parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
        args = parser.parse_args()

        # cfg
        config_file = args.config                       # change the path of the model config file
        grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
        sam_checkpoint = args.sam_checkpoint
        image_path = args.input_image
        split = args.split
        openai_key = args.openai_key
        openai_proxy = args.openai_proxy
        output_dir = args.output_dir
        box_threshold = args.box_threshold
        text_threshold = args.box_threshold
        iou_threshold = args.iou_threshold
        device = args.device

        os.makedirs(output_dir, exist_ok=True)
        openai.api_key = openai_key
        if openai_proxy:
            openai.proxy = {"http": openai_proxy, "https": openai_proxy}
    
    image_pil, image = load_image(image_path)  # IMG (4032, 3024), image[3, 800, 1066]
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # BLIP: gen caption, Tag2Text gen better captions https://huggingface.co/spaces/xinyu1205/Tag2Text
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
    caption = generate_caption(image_pil) # BLIP
    print(f"Caption: {caption}")

    # ChatGPT: get tags.
    text_prompt = generate_tags(caption, split=split) 
    #text_prompt = 'the yellow stool'
    print(f"Tags: {text_prompt}")

    # GrDINO, SAM (text_prompt)
    GrDINO_SAM = True
    if GrDINO_SAM: 
        # GroundingDINO {boxes_filt, pred_phrases}
        model = load_model(config_file, grounded_checkpoint, device=device) 
        boxes_filt, scores, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device) 

        # SAM 
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        image = cv2.imread(image_path)   #(3024, 4032, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)): 
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2  
            boxes_filt[i][2:] += boxes_filt[i][:2]         
        boxes_filt = boxes_filt.cpu()   # norm-raw

        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist() 
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS:  {boxes_filt.shape[0]} boxes")

        # SAM(rawboxes[X,Y,X,Y], featboxes[X,Y,X,Y], raw-masks)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False) # [2, 1, 3024, 4032]

    # ChatGPT: updated caption.
    caption = check_caption(caption, pred_phrases)
    print(f"Revise caption with number: {caption}")
    breakpoint()

    # draw output image 
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.title(caption)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "automatic_label_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    save_mask_data(output_dir, caption, masks, boxes_filt, pred_phrases)


