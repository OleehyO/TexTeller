#!/usr/bin/env python3
import os
import argparse
import torch

from pathlib import Path
from PIL import Image
from .model.Resizer import Resizer
from .utils import preprocess_fn

from munch import Munch


def inference(args):
    img = Image.open(args.image)
    img = img.convert('RGB') if img.format == 'PNG' else img
    processed_img = preprocess_fn({"pixel_values": [img]})

    ckt_path = Path(args.checkpoint).resolve()
    model = Resizer.from_pretrained(ckt_path)
    model.eval()
    inpu = torch.stack(processed_img['pixel_values'])
    pred = model(inpu) * 1.25
    print(pred)

    ...


if __name__ == "__main__":
    cur_dirpath = os.getcwd()
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--image', type=str, required=True)
    parser.add_argument('-ckt', '--checkpoint', type=str, required=True)

    args = parser.parse_args([
        '-img', '/home/lhy/code/TeXify/src/models/resizer/foo5_140h.jpg',
        '-ckt', '/home/lhy/code/TeXify/src/models/resizer/train/train_result_pred_height_v5'
    ])
    inference(args)

    os.chdir(cur_dirpath)