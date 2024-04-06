import numpy as np
import re

from models.ocr_model.utils.inference import inference as latex_inference


def to_katex(formula: str) -> str:
    res = formula
    res = re.sub(r'\\mbox\{([^}]*)\}', r'\1', res)
    res = re.sub(r'boldmath\$(.*?)\$', r'bm{\1}', res)
    res = re.sub(r'\\\[(.*?)\\\]', r'\1\\newline', res) 

    pattern = r'(\\(?:left|middle|right|big|Big|bigg|Bigg|bigl|Bigl|biggl|Biggl|bigm|Bigm|biggm|Biggm|bigr|Bigr|biggr|Biggr))\{([^}]*)\}'
    replacement = r'\1\2'
    res = re.sub(pattern, replacement, res)
    if res.endswith(r'\newline'):
        res = res[:-8]
    return res


def load_lang_models(language: str):
    ...
    # language: 'ch' or 'en'
    # return det_model, rec_model (or model)


def load_det_tex_model():
    ...
    # return the loaded latex detection model


def mix_inference(latex_det_model, latex_rec_model, lang_model, img: np.ndarray, use_cuda: bool) -> str:
    ...
    # latex_inference(...)
