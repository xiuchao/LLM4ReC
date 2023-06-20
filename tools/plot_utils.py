import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


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
        sans16  =  ImageFont.truetype (fontPath, 12)
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

