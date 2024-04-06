import os
import argparse
import cv2 as cv

from pathlib import Path
from utils import to_katex
from models.ocr_model.utils.inference import inference as latex_inference
from models.ocr_model.model.TexTeller import TexTeller


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-img', 
        type=str, 
        required=True,
        help='path to the input image'
    )
    parser.add_argument(
        '-cuda', 
        default=False,
        action='store_true',
        help='use cuda or not'
    )
    args = parser.parse_args()

    # You can use your own checkpoint and tokenizer path.
    print('Loading model and tokenizer...')
    latex_rec_model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()
    print('Model and tokenizer loaded.')

    img = cv.imread(args.img)
    print('Inference...')
    res = latex_inference(latex_rec_model, tokenizer, [img], args.cuda)
    res = to_katex(res[0])
    print(res)
