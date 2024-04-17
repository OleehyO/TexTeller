import torch
import numpy as np

from transformers import RobertaTokenizerFast, GenerationConfig
from typing import List, Union

from models.ocr_model.model.TexTeller import TexTeller
from models.ocr_model.utils.transforms import inference_transform
from models.ocr_model.utils.helpers import convert2rgb
from models.globals import MAX_TOKEN_SIZE


def inference(
    model: TexTeller, 
    tokenizer: RobertaTokenizerFast,
    imgs_path: Union[List[str], List[np.ndarray]], 
    inf_mode: str = 'cpu',
    num_beams: int = 1,
) -> List[str]:
    model.eval()
    if isinstance(imgs[0], str):
        imgs = convert2rgb(imgs) 
    else:  # already numpy array(rgb format)
        assert isinstance(imgs[0], np.ndarray)
        imgs = imgs 
    imgs = inference_transform(imgs)
    pixel_values = torch.stack(imgs)

    model = model.to(inf_mode)
    pixel_values = pixel_values.to(inf_mode)

    generate_config = GenerationConfig(
        max_new_tokens=MAX_TOKEN_SIZE,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    pred = model.generate(pixel_values, generation_config=generate_config)
    res = tokenizer.batch_decode(pred, skip_special_tokens=True)
    return res
