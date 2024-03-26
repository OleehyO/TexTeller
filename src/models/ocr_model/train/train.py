import os

from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig

from .training_args import CONFIG
from ..model.TexTeller import TexTeller
from ..utils.functional import tokenize_fn, collate_fn, img_transform_fn, filter_fn
from ..utils.metrics import bleu_metric
from ...globals import MAX_TOKEN_SIZE, MIN_WIDTH, MIN_HEIGHT   


def train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer):
    training_args = TrainingArguments(**CONFIG)
    debug_mode = False
    if debug_mode:
        training_args.auto_find_batch_size = False
        training_args.num_train_epochs = 2
        # training_args.per_device_train_batch_size = 3
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 2 * training_args.per_device_train_batch_size
        training_args.jit_mode_eval = False
        training_args.torch_compile = False
        training_args.dataloader_num_workers = 1
    
    trainer = Trainer(
        model,
        training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        tokenizer=tokenizer, 
        data_collator=collate_fn_with_tokenizer,
    )

    trainer.train(resume_from_checkpoint=None)
    # trainer.train(resume_from_checkpoint='/home/lhy/code/TexTeller/src/models/ocr_model/train/train_result/TexTellerv2/checkpoint-288000')


def evaluate(model, tokenizer, eval_dataset, collate_fn):
    eval_config = CONFIG.copy()
    eval_config['predict_with_generate'] = True
    generate_config = GenerationConfig(
        max_length=MAX_TOKEN_SIZE-100,
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    eval_config['generation_config'] = generate_config
    eval_config['auto_find_batch_size'] = False
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
    print(res)
    

if __name__ == '__main__':
    cur_path = os.getcwd()
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    dataset = load_dataset(
        '/home/lhy/code/TexTeller/src/models/ocr_model/train/data/loader.py'
    )['train']
    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TexTeller/src/models/tokenizer/roberta-tokenizer-7Mformulas')
    filter_fn_with_tokenizer = partial(filter_fn, tokenizer=tokenizer)

    # dataset = dataset.filter(lambda x: x['image'].height > MIN_HEIGHT and x['image'].width > MIN_WIDTH)
    dataset = dataset.filter(filter_fn_with_tokenizer, num_proc=16)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices()

    map_fn = partial(tokenize_fn, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(map_fn, batched=True, remove_columns=dataset.column_names, num_proc=8, load_from_cache_file=True)
    tokenized_dataset = tokenized_dataset.with_transform(img_transform_fn)

    split_dataset = tokenized_dataset.train_test_split(test_size=0.005, seed=42)
    train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    # model = TexTeller()
    model = TexTeller.from_pretrained('/home/lhy/code/TexTeller/src/models/ocr_model/model/ckpt')

    enable_train    = True
    enable_evaluate = True
    if enable_train:
        train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer)
    if enable_evaluate:
        evaluate(model, tokenizer, eval_dataset, collate_fn_with_tokenizer)

    os.chdir(cur_path)
