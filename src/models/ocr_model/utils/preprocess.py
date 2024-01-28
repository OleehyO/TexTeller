import torch
import datasets

from datasets import load_dataset

from functools import partial
from transformers import DataCollatorForLanguageModeling
from typing import List, Dict, Any
from ...ocr_model.model.TexTeller import TexTeller
from .transforms import train_transform


def left_move(x: torch.Tensor, pad_val):
    assert len(x.shape) == 2, 'x should be 2-dimensional'
    lefted_x = torch.ones_like(x)
    lefted_x[:, :-1] = x[:, 1:]
    lefted_x[:, -1] = pad_val
    return lefted_x


def tokenize_fn(samples: Dict[str, List[Any]], tokenizer=None) -> Dict[str, List[Any]]:
    assert tokenizer is not None, 'tokenizer should not be None'
    tokenized_formula = tokenizer(samples['latex_formula'], return_special_tokens_mask=True)
    tokenized_formula['pixel_values'] = samples['image']
    return tokenized_formula


def collate_fn(samples: List[Dict[str, Any]], tokenizer=None) -> Dict[str, List[Any]]:
    assert tokenizer is not None, 'tokenizer should not be None'
    pixel_values = [dic.pop('pixel_values') for dic in samples]

    clm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    batch = clm_collator(samples)
    batch['pixel_values'] = pixel_values
    batch['decoder_input_ids'] = batch.pop('input_ids')
    batch['decoder_attention_mask'] = batch.pop('attention_mask')

    # 左移labels和decoder_attention_mask
    batch['labels'] = left_move(batch['labels'], -100)
    batch['decoder_attention_mask'] = left_move(batch['decoder_attention_mask'], 0)

    # 把list of Image转成一个tensor with (B, C, H, W)
    batch['pixel_values'] = torch.stack(batch['pixel_values'], dim=0)
    return batch


def img_preprocess(samples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    processed_img = train_transform(samples['pixel_values'])
    samples['pixel_values'] = processed_img
    return samples


if __name__ == '__main__':
    dataset = load_dataset(
        '/home/lhy/code/TeXify/src/models/ocr_model/train/dataset/latex-formulas/latex-formulas.py',
        'cleaned_formulas'
    )['train'].select(range(20))
    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550Kformulas')

    map_fn = partial(tokenize_fn, tokenizer=tokenizer)
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    tokenized_formula = dataset.map(map_fn, batched=True, remove_columns=dataset.column_names)
    tokenized_formula = tokenized_formula.to_dict()
    # tokenized_formula['pixel_values'] = dataset['image']
    # tokenized_formula = dataset.from_dict(tokenized_formula)
    tokenized_dataset = tokenized_formula.with_transform(img_preprocess)

    dataset_dict = tokenized_dataset[:]
    dataset_list = [dict(zip(dataset_dict.keys(), x)) for x in zip(*dataset_dict.values())]
    batch = collate_fn_with_tokenizer(dataset_list)

    from ..model.TexTeller import TexTeller
    model = TexTeller()
    out = model(**batch)

    pause = 1

