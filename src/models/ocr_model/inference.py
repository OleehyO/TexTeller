import torch
import cv2
import numpy as np

from transformers import RobertaTokenizerFast, GenerationConfig
from PIL import Image
from typing import List

from .model.TexTeller import TexTeller
from .utils.transforms import inference_transform
from .utils.helpers import convert2rgb
from ...globals import MAX_TOKEN_SIZE


def inference(model: TexTeller, imgs_path: List[str], tokenizer: RobertaTokenizerFast) -> List[str]:
    imgs = convert2rgb(imgs_path)
    imgs = inference_transform(imgs)
    pixel_values = torch.stack(imgs)

    generate_config = GenerationConfig(
        max_new_tokens=MAX_TOKEN_SIZE,
        num_beams=3,
        do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    pred = model.generate(pixel_values, generation_config=generate_config)
    res = tokenizer.batch_decode(pred, skip_special_tokens=True)
    return res


if __name__ == '__main__':
    inference()
