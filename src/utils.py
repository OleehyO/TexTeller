import numpy as np

from models.ocr_model.utils.inference import inference as latex_inference


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
