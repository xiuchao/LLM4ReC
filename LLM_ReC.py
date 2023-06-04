import argparse
import os
import re
import copy
import json

import torch
import torchvision
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import gradio as gr

# ChatGPT, BLIP
import openai
from transformers import BlipProcessor, BlipForConditionalGeneration

# CLIP
import clip

# GrDINO, SAM
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 

# langchain
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

# utils
class PlotUtils:
    def __init__(self):
        self.background_fill = 0
    
    @staticmethod
    def _plt_show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def _plt_show_box(box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label)

    @staticmethod
    def plot_image_boxes_masks(output_dir, image_path, prompt_boxes, masks, pred_phrases, title="", opt=['masks', 'boxes']):
        assert len(opt) > 0, "opt must be a list of 'masks', 'boxes' or both"
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))

        image = cv2.imread(image_path)   #(3024, 4032, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        plt.imshow(image)

        if 'masks' in opt:
            for mask in masks:
                PlotUtils._plt_show_box(mask.cpu().numpy(), plt.gca(), random_color=True)

        if 'boxes' in opt:
            for box, label in zip(prompt_boxes, pred_phrases):
                PlotUtils._plt_show_mask(box.numpy(), plt.gca(), label)

        plt.title("")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "raw_box_mask.jpg"),  bbox_inches="tight", dpi=300, pad_inches=0.0)
    
def draw_candidate_boxes(image_path, crops_base_list, output_dir, stepstr= 'targets', save=False):
    #assert stepstr in ['candidates', 'self', 'related', 'ref'], "stepstr must be one of ['self', 'related', 'ref']"
    image_pil = Image.open(image_path).convert("RGB") 
    H, W = image_pil.size
    boxes =  [k["box"]  for k in crops_base_list]
    labels = [k["name"] for k in crops_base_list]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    #mask = Image.new("L", image_pil.size, 0)  #box mask
    #mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # draw rectangle
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=8)

        # draw textbox+text.
        fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
        sans16  =  ImageFont.truetype ( fontPath, 48 )
        font = sans16  #font = ImageFont.load_default()
        label_txt = label[0]+"("+str(label[1])+")"
        if hasattr(font, "getbbox"):
            txtbox = draw.textbbox((x0, y0), label_txt, font)
        else:
            w, h = draw.textsize(label_txt, font)
            txtbox = (x0, y0, w + x0, y0 + h)

        draw.rectangle(txtbox, fill=color)
        draw.text((x0, y0), label_txt, fill="white", font=font)
        #mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=8)

    if save:
        image_pil.save(os.path.join(output_dir, stepstr+".jpg")) 
    return image_pil

def draw_overlay_caption(image_path, request, withcaption=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(Image.open(os.path.join(image_path, "gptref.jpg")))
    plt.title(request)
    plt.axis('off')
    plt.savefig(os.path.join(image_path, "gptref_title.jpg"),  bbox_inches="tight", dpi=300, pad_inches=0.0)
    return True

class GroundedDetection:
    # GroundingDino
    def __init__(self, cfg):
        print(f"Initializing GroundingDINO to {cfg.device}")
        self.model = build_model(SLConfig.fromfile('GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'))
        checkpoint = torch.load('groundingdino_swint_ogc.pth', map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        self.processor = T.Compose([ 
                            T.RandomResize([800], max_size=1333),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.cfg = cfg

    def inference(self, image_path, caption, box_threshold, text_threshold, iou_threshold):
        self.model = self.model.to(self.cfg.device)

        # input: image, caption
        image_pil = Image.open(image_path).convert("RGB")  # load image
        image, _ = self.processor(image_pil, None) 
        image = image.to(self.cfg.device)

        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        # output: (boxes_filt, scores, pred_phrases)
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]    # num_filt, 4
        logits_filt.shape[0]

        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        # postprocessing: norm2raw: xywh2xyxy
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]) 
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2  
            boxes_filt[i][2:] += boxes_filt[i][:2]         
        boxes_filt = boxes_filt.cpu()  # norm2raw: xywh2xyxy

        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, torch.tensor(scores), iou_threshold).numpy().tolist() 
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS:  {boxes_filt.shape[0]} boxes")

        return boxes_filt, pred_phrases

class DetPromptedSegmentation:
    # SAM
    def __init__(self, cfg):
        self.predictor = SamPredictor(build_sam(checkpoint='sam_vit_h_4b8939.pth'))
        self.cfg = cfg

    @staticmethod
    def save_mask_json(output_dir, mask_list, box_list, label_list, caption=''):
        value = 0  # 0 for white background

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
                'box': [int(x) for x in box.numpy().tolist()],
            })
        with open(os.path.join(output_dir, 'label.json'), 'w') as f:
            json.dump(json_data, f)
        return json_data

    def inference(self, image_path, prompt_boxes, save_json=False):
        image = cv2.imread(image_path)   #(3024, 4032, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        self.predictor.set_image(image)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(prompt_boxes, image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False) # masks [n, 1, H, W], boxes_filt [n, 4]
        
        # plot_raw_boxes_masks(image, boxes_filt, masks, pred_phrases)
        if save_json == True:
            DetPromptedSegmentation.save_mask_json(self.cfg.output_dir, masks, prompt_boxes, pred_phrases)

        return masks

class ImageCropsBaseAttributesMatching:
    # CLIP 
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.preprocess = clip.load('ViT-B/32', self.cfg.device)
        self.user_base_attributes = self.user_defined_base_attributes()
        self.clip_text_base_attribute_embedding()
        
    
    def user_defined_base_attributes(self):
        base_attributes = {
            'color': [tuple(['blue', 'green', 'yellow', 'orange', 'red', 'white',  'violet', 'brown', 'aqua', 'black', 'cyan', 'purple']), None],
            'shape': [tuple(['rectangle', 'oval', 'star', 'cylinder', 'cube', 'pyramid', 'cone', 'sphere', 'cuboid']), None],
            'texture': [tuple(['metal', 'wood', 'leather', 'glass', 'plastic', 'ceramic', 'fabric', 'paper']), None], }
        return base_attributes

    def clip_text_base_attribute_embedding(self):
        base_dict_pth = os.path.join(self.cfg.output_dir, 'base_attr_embs.pth')
        if os.path.exists(base_dict_pth):
            self.user_base_embeddings = torch.load(base_dict_pth)
        else:
            self.user_base_embeddings = self.user_base_attributes.copy()
            for attr, val_list in self.user_base_attributes.items():
                embeddings = []
                for c in val_list[0]:
                    token_inputs = torch.cat([clip.tokenize(f"{c}") ]).to(self.cfg.device)        # [12, 77]
                    token_features = self.model.encode_text(token_inputs)                # [1, 512]
                    token_features /= token_features.norm(dim=-1, keepdim=True)          # [1, 512]
                    embeddings.append(token_features)                                    # [Ntokens, 512]
                embeddings = torch.cat(embeddings, 0).cpu()
                self.user_base_embeddings[attr][1] = embeddings
            torch.save(self.user_base_embeddings, base_dict_pth)
            print("---> save attribute emdeddings done!")

    def clip_image_crops_embedding(self, image_pil, boxes, masks):
        image_crop_features = []
        for i in range(boxes.shape[0]):
            # mask-crops
            mask = masks[i,0,].int().to(torch.uint8).numpy() 
            image_masked = np.array(image_pil)
            image_masked[mask==0] =255    # white
            image_pil_masked = Image.fromarray(image_masked)
            image_crops = image_pil_masked.crop(boxes.numpy()[i,]) 

            # encode
            image_input = self.preprocess(image_crops)         # [1, 3, 224, 224]
            image_input = image_input.unsqueeze(0).to(self.cfg.device)  # [1, 3, 224, 224]
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)           # [1, 512]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_crop_features.append(image_features)                      
        return image_crop_features  # [Ncrops, 512]   

    def get_objects_base_attributes(self, image_pil, boxes, pred_phrases, masks):
        crop_features = self.clip_image_crops_embedding(image_pil, boxes, masks)

        objects_base_attributes = [] 
        for i in range(boxes.shape[0]):
            base_attributes = {'name':['', None],
                            'box': None, 'width': None, 'height': None, 
                            'color':['', None], 
                            'shape':['', None], 
                            'texture':['',None]} 
            # detect crops 
            name_, score_ = re.match(r'(\w+)\((.*?)\)', pred_phrases[i]).groups()
            base_attributes ['name'] = [name_, float(score_)]
            base_attributes ['box'] = pos = [int(x) for x in boxes.numpy()[i,].tolist()]
            base_attributes ['width']  = pos[2] - pos[0]
            base_attributes ['height'] = pos[3] - pos[1]


            # match (crop, base-dict)
            for attr, val_list in self.user_base_embeddings.items():
                text_features = val_list[1].to(self.cfg.device)
                similarity = (100.0 * crop_features[i] @ text_features.T).softmax(dim=-1) 
                base_attributes[attr] = val_list[0][similarity.argmax()], similarity.max().detach().cpu().numpy().item()
            objects_base_attributes.append(base_attributes)

        return objects_base_attributes

class ImageBLIPCaptioning:
    # BLIP
    def __init__(self): 
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model =  BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        self.model.eval()
    
    def inference(image):
        inputs =self.processor(raw_image, return_tensors="pt").to("cuda", torch.float16) 
        out = self.model.generate(**inputs) 
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

        # BLIP crop caption
        # crop_caption = BLIP.inference(image_crop)
        # print(f"Crop caption {i}: {crop_caption}")

class GPT4Reasoning: 
    def __init__(self):
        self.llms = ["gpt-3.5-turbo", "gpt-4"]
        self.split=','  
        self.max_tokens=100 
        self.temperature=0.6
        self.prompt = [{ 'role': 'system', 'content': ''}]

    def extract_unique_nouns(self, request): 
        self.prompt[0]['content'] = 'Extract the unique objects in the caption. Remove all the adjectives.' + \
                        f'List the nouns in singular form. Split them by "{self.split} ". ' + \
                        f'Caption: {request}.'

        response = openai.ChatCompletion.create(model=self.llms[0], messages=self.prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        reply = response['choices'][0]['message']['content']
        unique_nouns = reply.split(':')[-1].strip() # sometimes return with "noun: xxx, xxx, xxx"
        return unique_nouns

    # reasoning related objects: subgraph_dict
    def extract_target_relations(self, request): 
        # Request: "give the red apple between the blue cup and round white plate."
        # Request: "give the red apple and yellow apple between the blue cup and the round white plate.
        self.prompt[0]['content'] = 'Extract the target object, related objects and spatial relationships from the human request.' + \
                'Specify the object in singular mode, e.g. two red apples as (red apple, red apple)' + \
                'Output follow standard JSON string format.Here is an example:' + \
                'request: give the red cup next to the red apple, the spoon on the left of the square plate.' + \
                '{  "target":  [{"name": "cup", "color": "red"}, {"name": "spoon"}],' + \
                '   "related": [{"spatial": "next to", "target":"0", "name": "apple", "color": "red"},' + \
                '               {"spatial": "on the left of", "target":"1", "name": "plate", "shape": "square"}]}' + \
                f'Human request: {request}.'

        response = openai.ChatCompletion.create(model=self.llms[1], messages=self.prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        reply = response['choices'][0]['message']['content']
        def json2dict(input_str):
            input_str = input_str.replace("'", '"').strip()
            data = json.loads(input_str)
            return data
        subgraph_dict = json2dict(reply)
        return subgraph_dict

    # locate target, disambiguation
    def extract_disambiguated_target(self, request, crops_attributes_list, image_size=(None, None)):
        self.prompt[0]['content'] = 'Locate the target object(s) in the request according to the given information.' + \
                'The image size is {image_size}. Output only the order index, like 0, 1, 2...' + \
                f'Human request: {request}.' + \
                f'Candidate objects: {crops_attributes_list}.'

        response = openai.ChatCompletion.create(model=self.llms[1], messages=self.prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        reply = response['choices'][0]['message']['content']
        return reply 

    def get_counted_BLIP(self, caption, pred_phrases):
        object_list = [obj.split('(')[0] for obj in pred_phrases]
        object_num = []
        for obj in set(object_list):
            object_num.append(f'{object_list.count(obj)} {obj}')
        object_num = ', '.join(object_num)
        print(f"Correct object number: {object_num}")

        self.prompt[0]['content'] = 'Revise the number in the caption if it is wrong. ' + \
                        f'Caption: {caption}. ' + \
                        f'True object number: {object_num}. ' + \
                        'Only give the revised caption. ' 
        
        response = openai.ChatCompletion.create(model=self.llms[1], messages=self.prompt,temperature=self.temperature, max_tokens=self.max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "Caption: xxx, xxx, xxx"
        caption = reply.split(':')[-1].strip()
        return caption

class TargetMatching:
    def __init__(self, image_path, request, crops_base_list, text_gpt_dict, cfg):
        self.rawimage = image_path
        image_pil = Image.open(image_path).convert("RGB") 
        self.imgsize = image_pil.size
        self.request = request
        self.subgraph = text_gpt_dict
        self.count = len(crops_base_list)
        self.crops_base_list = crops_base_list
        self.cfg = cfg

    def match_subgraph_target_base(self):
        subgraph_target_list = []
        for tgt in self.subgraph['target']:
            keys_check = [k for k in tgt.keys() if k in self.crops_base_list[0].keys()]
            subgraph_target_list = [c for c in self.crops_base_list if all(c[k][0] == tgt[k] and c[k][1] >= 0.3 for k in keys_check)]
        print(f'{len(subgraph_target_list)} boxes found!')
        
        self.count = len(subgraph_target_list)
        if self.cfg.visualize:
            draw_candidate_boxes(self.rawimage, self.crops_base_list, cfg.output_dir, stepstr='nouns', save=True)
        return subgraph_target_list

    def match_subgraph_related_base(self, subgraph_target_list):
        assert len(subgraph_target_list) > 1, "No related objects in subgraph!"
        print('get boxes for disambiguation!')
        subgraph_target_related_list = subgraph_target_list
        for item in self.subgraph['related']:
            keys_check = [k for k in item.keys() if k in subgraph_target_list[0].keys()]
            subgraph_target_related_list.append([c for c in subgraph_target_list if all(c[k][0] == item[k] and c[k][1] >= 0.3 for k in keys_check)])

        subgraph_target_related_list = [ x for x in subgraph_target_related_list if x != [] ]
        self.count = len(subgraph_target_related_list)
        if cfg.visualize:
            draw_candidate_boxes(self.rawimage, subgraph_target_related_list, self.cfg.output_dir, stepstr= 'related', save=True)
        return subgraph_target_related_list

    def match_subgraph_reasoning_base(self,subgraph_list):
        # gpt for disambiguation
        gpt_ref_ids = GPT4Reasoning().extract_disambiguated_target(self.request, subgraph_list, self.imgsize)
        ref_ids = list(map(int, gpt_ref_ids.split(',')))
        gpt_ref_list = [subgraph_list[i] for i in ref_ids]

        self.count = len(gpt_ref_list)
        if cfg.visualize:
            draw_candidate_boxes(self.rawimage, gpt_ref_list, self.cfg.output_dir, stepstr= 'gptref', save=True)
            draw_overlay_caption(self.cfg.output_dir, self.request, withcaption=True)
        return gpt_ref_list
    
    def match_complex_attributes(self):
        # "add function to match complex attributes"
        CropBLIP = False
        if CropBLIP:
            image_crop.save(os.path.join(output_dir, "crop_"+str(i)+ ".jpg"))
            # BLIP crop caption
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
            crop_caption = generate_caption(image_crop) # BLIP
            print(f"Crop caption {i}: {crop_caption}")
        pass

    def inference(self):
        # matching sequence
        subgraph_target_list = self.match_subgraph_target_base()
        if self.count > 1:
            subgraph_related_list = self.match_subgraph_related_base(subgraph_target_list)
            if self.count > 1:
                gpt_ref_list = self.match_subgraph_reasoning_base(subgraph_related_list)
                print(f'{self.count} targets found!')
                return gpt_ref_list
            elif self.count == 1:
                return subgraph_related_list
            else:
                return None
        elif self.count == 1:
            print('target object found!')
            return subgraph_target_list
        else:
            print('no target object found!')
            return None

# image: objects_base_attributes [{}]
def img_inference_grounded_objects_base_attributes(image_path, request, cfg):
    image_pil = Image.open(image_path).convert("RGB") 
    unique_nouns = GPT4Reasoning().extract_unique_nouns(request) 

    detector = GroundedDetection(cfg)
    boxes, pred_phrases = detector.inference(image_path, unique_nouns, cfg.box_threshold, cfg.text_threshold, cfg.iou_threshold)
    segmenter = DetPromptedSegmentation(cfg)
    masks = segmenter.inference(image_path, prompt_boxes=boxes, save_json=False)

    matcher_base = ImageCropsBaseAttributesMatching(cfg)
    objects_base_attributes = matcher_base.get_objects_base_attributes(image_pil, boxes, pred_phrases, masks) 
    return objects_base_attributes

# text: subgraph_base_attributes {k:[{},{}]}
def txt_inference_subgraph(request):
    # ----------------- subgraph_base_attributes -----------------
    #   example for single target test: "give me the yellow stool"
    #   subgraph_base_attributes = {'target': [{'name': 'stool', 'color': 'yellow'}], 'related': []}  
    #  
    #   example for s single target, related with other objects.
    #   subgraph_base_attributes = { 'target':  [{'name': 'stool', 'color': 'red'}], 
    #                              'related': [{'spatial': 'on the left of', 'target': '0', 'name': 'stool', 'color': 'yellow'}]}
    subgraph_dict = GPT4Reasoning().extract_target_relations(request) 
    return subgraph_dict

def main(image_path, request, cfg):
    # input inference
    crops_base_list = img_inference_grounded_objects_base_attributes(image_path, request, cfg)
    text_subgraph = txt_inference_subgraph(request)

    # matching process
    chatref = TargetMatching(image_path, request, crops_base_list, text_subgraph, cfg)
    targets = chatref.inference()


# gradio bot
class ConversationBot:
    def __init__(self, load_dict):
        print(f"Initializing LAA")
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

        self.tools = [] 
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 
                          'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                          'suffix': VISUAL_CHATGPT_SUFFIX}, )

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_image(self, image, state, txt):
        # input
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        
        # do the job
        description = self.models['ImageCaptioning'].inference(image_filename)
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} '


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser("ChatRef", add_help=True)
    parser.add_argument("--openai_key", type=str, required=True, help="openai key")
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--request", type=str, default="give me something", required=True, help="human request")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--visualize", default=False, help="visualize intermediate data mode")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    cfg = parser.parse_args()

    image_path = cfg.input_image
    request = cfg.request
    cfg.output_dir = os.path.join(cfg.output_dir, request)
    os.makedirs(cfg.output_dir,exist_ok=True)

    openai.api_key = cfg.openai_key
    openai_proxy = None
    if openai_proxy:
        openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # main()
    main(image_path, request, cfg)
    if False:
        bot = ConversationBot(load_dict=load_dict)
        with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
            chatbot = gr.Chatbot(elem_id="chatbot", label="ChatRef")
            state = gr.State([])
            with gr.Row():
                with gr.Column(scale=0.7):
                    txt = gr.Textbox(show_label=False, placeholder="Upload an image, Enter text request, Press Enter").style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("Clear")
                with gr.Column(scale=0.15, min_width=0):
                    btn = gr.UploadButton("Upload", file_types=["image"])

            txt.submit(bot.run_text, [txt, state], [chatbot, state])
            txt.submit(lambda: "", None, txt)
            btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
            clear.click(bot.memory.clear)
            clear.click(lambda: [], None, chatbot)
            clear.click(lambda: [], None, state)
            demo.launch(server_name="0.0.0.0", server_port=7999)

