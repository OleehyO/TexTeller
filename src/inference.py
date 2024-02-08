import os
import argparse

from pathlib import Path
from models.ocr_model.utils.inference import inference
from models.ocr_model.model.TexTeller import TexTeller


if __name__ == '__main__':
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

    args = parser.parse_args([
        '-img', './models/ocr_model/test_img/1.png',
        '-cuda'
    ])

    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    model = TexTeller.from_pretrained('./models/ocr_model/model_checkpoint')
    tokenizer = TexTeller.get_tokenizer('./models/tokenizer/roberta-tokenizer-550K')

    # base = '/home/lhy/code/TeXify/src/models/ocr_model/test_img'
    # img_path = [base + f'/{i}.png' for i in range(7, 12)]
    img_path = [args.img]

    res = inference(model, tokenizer, img_path, args.cuda)
    print(res[0])
