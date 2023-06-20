import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageBLIPCaptioning:
    # BLIP
    def __init__(self): 
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model =  BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        self.model.eval()
    
    def inference(self, image):
        inputs =self.processor(image, return_tensors="pt").to("cuda", torch.float16) 
        out = self.model.generate(**inputs) 
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

        # BLIP crop caption
        # crop_caption = BLIP.inference(image_crop)
        # print(f"Crop caption {i}: {crop_caption}")

