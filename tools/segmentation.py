import os
import json
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything.segment_anything import build_sam, SamPredictor


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
            DetPromptedSegmentation.save_mask_json(self.cfg.output_dir, masks, prompt_boxes)

        return masks

