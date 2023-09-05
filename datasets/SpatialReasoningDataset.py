import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SpatialReasoningDataset(Dataset):
    def __init__(self, version="refcoco_unc", 
                 split="testA", 
                 im_dir="data/refcoco/images/train2014",
                 ann_root="data/refcoco/anns_spatial/"):
        self.version = version
        self.split = split
        self.im_dir = im_dir
        self.ann_root = ann_root
        self.anns = self._load_annotations()
        self.idx_to_iname = self.get_idx_to_image_name()
        self.data = self.get_phrase_and_image_info()
        #TODO: load bbox info of each image


    def _load_annotations(self):
        dname = self.version.split("_")[0]
        ann_dir = os.path.join(self.ann_root, dname)
        ann_name = f"{self.version}_{self.split}_dict.pth"
        ann_file = os.path.join(ann_dir, ann_name)
        anns = torch.load(ann_file)
        return anns
    

    def get_idx_to_image_name(self):
        anns = []
        for k, v in self.anns.items():
            v["filename"] = k
            anns.append(v)
        idx_to_iname = {}
        for i, d in enumerate(anns):
            idx_to_iname[i] = d["filename"]
        return idx_to_iname
    

    def get_phrase_and_image_info(self):
        data = []
        for k, v in self.anns.items():
            image_path = os.path.join(self.im_dir, k)
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            boxes_and_phrases = v["box"] # dict
            mask_files = v["mask"]
            i = 0
            for box_tuple, phrases in boxes_and_phrases.items():
                box_round = [np.round(x, 2) for x in box_tuple]
                mask_file = mask_files[i]
                i += 1
                for phrase in phrases:
                    item = {
                        "phrase": phrase,
                        "image_name": k,
                        "image_size": [w, h],
                        "target_box": box_round,
                        "mask_file": mask_file
                    }
                    data.append(item)
        return data
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        data = self.data[index]
        return data


if __name__ == "__main__":
    dataset = SpatialReasoningDataset()
