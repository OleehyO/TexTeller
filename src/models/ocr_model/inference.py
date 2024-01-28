import torch
from transformers import RobertaTokenizerFast, GenerationConfig
from PIL import Image
from typing import List

from .model.TexTeller import TexTeller
from .utils.transforms import inference_transform
from ...globals import MAX_TOKEN_SIZE


def png2jpg(imgs: List[Image.Image]):
    imgs = [img.convert('RGB') for img in imgs if img.mode in ("RGBA", "P")]
    return imgs


def inference(model: TexTeller, imgs: List[Image.Image], tokenizer: RobertaTokenizerFast) -> List[str]:
    imgs = png2jpg(imgs) if imgs[0].mode in ('RGBA' ,'P') else imgs
    imgs = inference_transform(imgs)
    pixel_values = torch.stack(imgs)

    generate_config = GenerationConfig(
        max_new_tokens=MAX_TOKEN_SIZE,
        num_beams=3,
        do_sample=False
    )
    pred = model.generate(pixel_values, generation_config=generate_config)
    res = tokenizer.batch_decode(pred, skip_special_tokens=True)
    return res


if __name__ == '__main__':
    inference()
