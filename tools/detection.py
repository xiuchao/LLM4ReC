from PIL import Image
import torch
import torchvision
# GrDINO, SAM
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


class GroundedDetection:
    # GroundingDino
    def __init__(self, cfg):
        self.cfg = cfg
        print(f"Initializing GroundingDINO to {cfg.device}")
        self.model = build_model(SLConfig.fromfile('GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'))
        checkpoint = torch.load('groundingdino_swint_ogc.pth', map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.to(self.cfg.device)
        self.model.eval()
        self.processor = T.Compose([ 
                            T.RandomResize([800], max_size=1333),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def inference(self, image_pil, caption, box_threshold, text_threshold, iou_threshold):
        # input: image, caption
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

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]    # num_filt, 4

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

