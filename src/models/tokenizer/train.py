import os
from pathlib import Path
from datasets import load_dataset
from ..ocr_model.model.TexTeller import TexTeller
from ..globals import VOCAB_SIZE


if __name__ == '__main__':
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    tokenizer = TexTeller.get_tokenizer()

    # Don't forget to config your dataset path in loader.py
    dataset = load_dataset('../ocr_model/train/dataset/loader.py')['train']

    new_tokenizer = tokenizer.train_new_from_iterator(
        text_iterator=dataset['latex_formula'], 

        # If you want to use a different vocab size, **change VOCAB_SIZE from globals.py**
        vocab_size=VOCAB_SIZE  
    )

    # Save the new tokenizer for later training and inference
    new_tokenizer.save_pretrained('./your_dir_name')
