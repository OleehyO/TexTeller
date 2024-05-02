import argparse

from models.ocr_model.model.TexTeller import TexTeller


parser = argparse.ArgumentParser()
parser.add_argument(
    '--ckpt-dir', 
    type=str, 
    required=True,
)


args = parser.parse_args()
ckpt_dir = args.ckpt_dir
model = TexTeller.from_pretrained(ckpt_dir)
model.save_pretrained(ckpt_dir, safe_serialization=False)
