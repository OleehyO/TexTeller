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
        '-img_dir',
        type=str,
        default="./subimages",
        help='path to the directory containing input images'
    )
    parser.add_argument(
        '-output_dir',
        type=str,
        default="./results",
        help='path to the output directory for storing recognition results'
    )
    parser.add_argument(
        '-cuda',
        default=False,
        action='store_true',
        help='use cuda or not'
    )

    args = parser.parse_args()

    print('Loading model and tokenizer...')
    latex_rec_model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()
    print('Model and tokenizer loaded.')

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Loop through all images in the input directory
    for filename in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, filename)
        img = cv.imread(img_path)

        if img is not None:
            print(f'Inference for {filename}...')
            res = latex_inference(latex_rec_model, tokenizer, [img], args.cuda)
            res = to_katex(res[0])

            # Save the recognition result to a text file
            output_file = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '.txt')
            with open(output_file, 'w') as f:
                f.write(res)

            print(f'Result saved to {output_file}')
        else:
            print(f"Warning: Could not read image {img_path}. Skipping...")
