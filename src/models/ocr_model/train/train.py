import os
import datasets

from functools import partial
from pathlib import Path

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig

from .training_args import CONFIG
from ..model.TexTeller import TexTeller
from ..utils.functional import tokenize_fn, collate_fn, img_train_transform, img_inf_transform, filter_fn
from ..utils.metrics import bleu_metric
from ...globals import MAX_TOKEN_SIZE


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
    # trainer.train(resume_from_checkpoint=max_checkpoint_dir)
    # from scratch
    trainer.train(resume_from_checkpoint=None)


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
    print(script_dirpath)

    handwritten_online_dataset = load_dataset(
        '/data/lhy/train_data/TexTeller/loader.py',
        "handwritten_online",
        num_proc=64
    )['train']
    handwritten_nature_dataset = load_dataset(
        '/data/lhy/train_data/TexTeller/loader.py',
        "handwritten_nature",
        num_proc=64
    )['train']
    en_dataset = load_dataset(
        '/data/lhy/train_data/TexTeller/loader.py',
        "en_formulas",
        num_proc=64
    )['train']
    zh_dataset = load_dataset(
        '/data/lhy/train_data/TexTeller/loader.py',
        "zh_formulas",
        num_proc=64
    )['train']

    # load tokenizer
    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TexTeller/src/models/tokenizer/roberta-tokenizer-7Mformulas')
    # tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TexTeller/src/models/tokenizer/roberta-tokenizer-70M-en-zh')

    NUM_PROC = 64
    # filter
    filter_fn_with_tokenizer = partial(filter_fn, tokenizer=tokenizer)
    filter_dataset = lambda dataset: dataset.filter(
        filter_fn_with_tokenizer,
        num_proc=NUM_PROC,
    )
    handwritten_online_dataset: Dataset = filter_dataset(handwritten_online_dataset)
    handwritten_nature_dataset: Dataset = filter_dataset(handwritten_nature_dataset)
    en_dataset: Dataset                 = filter_dataset(en_dataset)
    zh_dataset: Dataset                 = filter_dataset(zh_dataset)

    # map
    map_fn = partial(tokenize_fn, tokenizer=tokenizer)
    map_dataset = lambda dataset: dataset.map(
        map_fn, batched=True, remove_columns=dataset.column_names,
        num_proc=NUM_PROC,
        load_from_cache_file=True
    )
    handwritten_online_dataset: Dataset = map_dataset(handwritten_online_dataset)
    handwritten_nature_dataset: Dataset = map_dataset(handwritten_nature_dataset)
    en_dataset: Dataset                 = map_dataset(en_dataset)
    zh_dataset: Dataset                 = map_dataset(zh_dataset)

    # enlarge the proportion of handwritten dataset in the whole dataset
    handwritten_dataset   = datasets.concatenate_datasets([handwritten_online_dataset, handwritten_nature_dataset])
    handwritten_datasetX5 = datasets.concatenate_datasets([handwritten_dataset] * 5)
    dataset_wo_en         = datasets.concatenate_datasets([en_dataset, zh_dataset, handwritten_datasetX5])

    # only use en_dataset as eval dataset
    split_en_dataset = en_dataset.train_test_split(test_size=0.0005, seed=42)


    train_dataset, eval_dataset = split_en_dataset['train'], split_en_dataset['test']
    # mix English train dataset with other dataset
    train_dataset = datasets.concatenate_datasets([train_dataset, dataset_wo_en])

    # shuffle
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset  = eval_dataset.shuffle(seed=42)

    # transform
    train_dataset = train_dataset.with_transform(img_train_transform)
    eval_dataset  = eval_dataset.with_transform(img_inf_transform)
    #################### debug #########################
    # foo = train_dataset[:50]
    # bar = eval_dataset[:5]
    #################### debug #########################

    # prepare for training
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    # model = TexTeller.from_pretrained(max_checkpoint_dir)
    # from scratch
    model = TexTeller()

    enable_train    = True
    enable_evaluate = True

    if enable_train:
        train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer)
    if enable_evaluate:
        evaluate(model, tokenizer, eval_dataset, collate_fn_with_tokenizer)

    os.chdir(cur_path)
