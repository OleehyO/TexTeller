import os
import argparse

from pathlib import Path
from models.ocr_model.utils.inference import inference
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
    model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()
    print('Model and tokenizer loaded.')

    img_path = [args.img]
    print('Inference...')
    res = inference(model, tokenizer, img_path, args.cuda)
    print(res[0])
