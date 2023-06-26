import os
import re
from PIL import Image
import numpy as np
import torch
import clip
from tools.GPTReasoning import GPT4Reasoning
from tools.blip import ImageBLIPCaptioning
from tools.utils import prompts, read_image_pil
from tools.plot_utils import draw_candidate_boxes, draw_overlay_caption


class TargetMatching:
    def __init__(self, image_input, request, crops_base_list, text_gpt_dict, cfg):
        image_pil = read_image_pil(image_input)
        self.rawimage = image_pil
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
            image_pil = draw_candidate_boxes(self.rawimage, self.crops_base_list, self.cfg.output_dir, stepstr='nouns', save=True)
        return subgraph_target_list, image_pil

    def match_subgraph_related_base(self, subgraph_target_list):
        assert len(subgraph_target_list) > 1, "No related objects in subgraph!"
        print('get boxes for disambiguation!')
        subgraph_target_related_list = subgraph_target_list
        for item in self.subgraph['related']:
            keys_check = [k for k in item.keys() if k in subgraph_target_list[0].keys()]
            subgraph_target_related_list.extend([c for c in self.crops_base_list if all(c[k][0] == item[k] and c[k][1] >= 0.3 for k in keys_check)])

        subgraph_target_related_list = [ x for x in subgraph_target_related_list if x != [] ]
        self.count = len(subgraph_target_related_list)
        image_pil = None
        if self.cfg.visualize:
            image_pil = draw_candidate_boxes(self.rawimage, subgraph_target_related_list, self.cfg.output_dir, stepstr='related', save=True)
        return subgraph_target_related_list, image_pil

    def match_subgraph_reasoning_base(self, subgraph_list, showtitle=False):
        # gpt for disambiguation
        gpt_ref_ids = GPT4Reasoning(temperature=0).extract_disambiguated_target(self.request, subgraph_list, self.imgsize)
        ref_ids = list(map(int, gpt_ref_ids.split(',')))
        gpt_ref_list = [subgraph_list[i] for i in ref_ids]
        self.count = len(gpt_ref_list)
        
        image_pil = None
        if self.cfg.visualize:
            image_pil = draw_candidate_boxes(self.rawimage, gpt_ref_list, self.cfg.output_dir, stepstr='gptref', save=True)
            if showtitle:
                image_pil = draw_overlay_caption(self.cfg.output_dir, self.request, withcaption=True)
        return gpt_ref_list, image_pil
    
    def match_complex_attributes(self,):
        # "add function to match complex attributes"
        CropBLIP = False
        if CropBLIP:
            blip = ImageBLIPCaptioning()
            for i in range(len(self.crops_base_list)):
                image_crop = self.crops_base_list[i]
                image_crop.save(os.path.join(self.cfg.output_dir, "crop_"+str(i)+ ".jpg"))
                # BLIP crop caption
                crop_caption = blip.inference(image_crop) # BLIP
                print(f"Crop caption {i}: {crop_caption}")


    @prompts(name="locate target object that match human request",
             description="useful when you try to extract the target object from the image that align with human request")
    def inference(self):
        # matching sequence
        image_pil = None
        subgraph_target_list, image_pil = self.match_subgraph_target_base()
        if self.count > 1:
            subgraph_related_list, image_pil = self.match_subgraph_related_base(subgraph_target_list)
            if self.count > 1:
                gpt_ref_list, image_pil = self.match_subgraph_reasoning_base(subgraph_related_list, showtitle=True)
                print(f'{self.count} targets found!')
                return gpt_ref_list, image_pil
            elif self.count == 1:
                return subgraph_related_list, image_pil
            else:
                return None, None
        elif self.count == 1:
            print('target object found!')
            return subgraph_target_list, image_pil
        else:
            print('no target object found!')
            return None, None


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
                base_attributes[attr] = val_list[0][similarity.argmax()], round(similarity.max().detach().cpu().numpy().item(), 2)
            objects_base_attributes.append(base_attributes)

        return objects_base_attributes

