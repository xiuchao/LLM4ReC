import numpy as np
from PIL import Image


# utils
def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator


def read_image_pil(image_input) -> Image.Image:
    if type(image_input) == str:
        image = Image.open(image_input).convert("RGB")
    elif type(image_input) == np.ndarray:
        image = Image.fromarray(image_input)
    else: # Image.Image
        image = image_input
    return image
