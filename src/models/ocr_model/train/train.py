import os
import numpy as np

from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig

from .training_args import CONFIG
from ..model.TexTeller import TexTeller
from ..utils.functional import tokenize_fn, collate_fn, img_transform_fn
from ..utils.metrics import bleu_metric
from ....globals import MAX_TOKEN_SIZE


def train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer):
    training_args = TrainingArguments(**CONFIG)
    trainer = Trainer(
        model,
        training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        tokenizer=tokenizer, 
        data_collator=collate_fn_with_tokenizer,
    )

    trainer.train(resume_from_checkpoint=None)


def evaluate(model, tokenizer, eval_dataset, collate_fn):
    eval_config = CONFIG.copy()
    generate_config = GenerationConfig(
        max_new_tokens=MAX_TOKEN_SIZE,
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    # eval_config['use_cpu'] = True
    eval_config['output_dir'] = 'debug_dir'
    eval_config['predict_with_generate'] = True
    eval_config['predict_with_generate'] = True
    eval_config['dataloader_num_workers'] = 1
    eval_config['jit_mode_eval'] = False
    eval_config['torch_compile'] = False
    eval_config['auto_find_batch_size'] = False
    eval_config['generation_config'] = generate_config
    seq2seq_config = Seq2SeqTrainingArguments(**eval_config)

    trainer = Seq2SeqTrainer(
        model,
        seq2seq_config,

        eval_dataset=eval_dataset,
        tokenizer=tokenizer, 
        data_collator=collate_fn,
        compute_metrics=partial(bleu_metric, tokenizer=tokenizer)
    )

    res = trainer.evaluate()
    pause = 1
    ...
    


if __name__ == '__main__':
    cur_path = os.getcwd()
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)


    # dataset = load_dataset(
    #     '/home/lhy/code/TeXify/src/models/ocr_model/train/dataset/latex-formulas/latex-formulas.py',
    #     'cleaned_formulas'
    # )['train']
    dataset = load_dataset(
        '/home/lhy/code/TeXify/src/models/ocr_model/train/dataset/latex-formulas/latex-formulas.py',
        'cleaned_formulas'
    )['train'].select(range(1000))

    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550Kformulas')

    map_fn = partial(tokenize_fn, tokenizer=tokenizer)
    # tokenized_dataset = dataset.map(map_fn, batched=True, remove_columns=dataset.column_names, num_proc=8, load_from_cache_file=False)
    tokenized_dataset = dataset.map(map_fn, batched=True, remove_columns=dataset.column_names, num_proc=1, load_from_cache_file=False)
    tokenized_dataset = tokenized_dataset.with_transform(img_transform_fn)

    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)    
    train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    # model = TexTeller()
    model = TexTeller.from_pretrained('/home/lhy/code/TeXify/src/models/ocr_model/train/train_result/checkpoint-80500')

    enable_train = False
    enable_evaluate = True
    if enable_train:
        train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer)  
    if enable_evaluate:
        evaluate(model, tokenizer, eval_dataset, collate_fn_with_tokenizer)


    os.chdir(cur_path)

