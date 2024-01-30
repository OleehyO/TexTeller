import torch
import cv2
import numpy as np

from transformers import RobertaTokenizerFast, GenerationConfig
from PIL import Image
from typing import List

from .model.TexTeller import TexTeller
from .utils.transforms import inference_transform
from ...globals import MAX_TOKEN_SIZE


def convert2rgb(image_paths: List[str]) -> List[Image.Image]:
    processed_images = []

    for path in image_paths:
        # 读取图片
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Image at {path} could not be read.")
            continue

        # 检查图片是否使用 uint16 类型
        if image.dtype == np.uint16:
            raise ValueError(f"Image at {path} is stored in uint16, which is not supported.")

        # 获取图片通道数
        channels = 1 if len(image.shape) == 2 else image.shape[2]

        # 如果是 RGBA (4通道), 转换为 RGB
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # 如果是 I 模式 (单通道灰度图), 转换为 RGB
        elif channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 如果是 BGR (3通道), 转换为 RGB
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_images.append(Image.fromarray(image))

    return processed_images


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
