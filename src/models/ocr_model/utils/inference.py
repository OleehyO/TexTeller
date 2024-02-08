import torch

from transformers import RobertaTokenizerFast, GenerationConfig
from typing import List

from models.ocr_model.model.TexTeller import TexTeller
from models.ocr_model.utils.transforms import inference_transform
from models.ocr_model.utils.helpers import convert2rgb
from models.globals import MAX_TOKEN_SIZE


def inference(
    model: TexTeller, 
    tokenizer: RobertaTokenizerFast,
    imgs_path: List[str], 
    use_cuda: bool,
    num_beams: int = 1,
) -> List[str]:
    model.eval()
    imgs = convert2rgb(imgs_path)
    imgs = inference_transform(imgs)
    pixel_values = torch.stack(imgs)

    if use_cuda:
        model = model.to('cuda')
        pixel_values = pixel_values.to('cuda')


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
