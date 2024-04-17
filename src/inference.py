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
        '--inference-mode', 
        type=str,
        default='cpu',
        help='Inference mode, select one of cpu, cuda, or mps'
    )
    parser.add_argument(
        '--num-beam', 
        type=int,
        default=1,
        help='number of beam search for decoding'
    )
    # =================  new feature  ==================
    parser.add_argument(
        '-mix', 
        type=str,
        help='use mix mode, only Chinese and English are supported.'
    )
    # ==================================================

    args = parser.parse_args()

    # You can use your own checkpoint and tokenizer path.
    print('Loading model and tokenizer...')
    latex_rec_model = TexTeller.from_pretrained()
    latex_rec_model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()
    print('Model and tokenizer loaded.')

    # img_path = [args.img]
    img = cv.imread(args.img)
    print('Inference...')
    if not args.mix:
        res = latex_inference(latex_rec_model, tokenizer, [img], args.inference_mode, args.num_beam)
        res = to_katex(res[0])
        print(res)
    else:
        # latex_det_model = load_det_tex_model()
        # lang_model      = load_lang_models()...
        ...
        # res: str = mix_inference(latex_det_model, latex_rec_model, lang_model, img, args.cuda)
        # print(res)
