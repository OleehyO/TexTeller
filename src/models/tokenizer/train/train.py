from datasets import load_dataset
from ...ocr_model.model.TexTeller import TexTeller
from ...globals import VOCAB_SIZE


if __name__ == '__main__':
    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TexTeller/src/models/tokenizer/roberta-tokenizer-raw')
    dataset = load_dataset("/home/lhy/code/TexTeller/src/models/ocr_model/train/data/loader.py")['train']
    new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=dataset['latex_formula'], vocab_size=VOCAB_SIZE)
    new_tokenizer.save_pretrained('/home/lhy/code/TexTeller/src/models/tokenizer/roberta-tokenizer-7Mformulas')
    pause = 1
