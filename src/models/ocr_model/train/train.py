import os

from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig

from .training_args import CONFIG
from ..model.TexTeller import TexTeller
from ..utils.functional import tokenize_fn, collate_fn, img_train_transform, img_inf_transform, filter_fn
from ..utils.metrics import bleu_metric
from ...globals import MAX_TOKEN_SIZE


def train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer):
    if MODEL_SIZE == 'small':
        # 4card
        CONFIG['output_dir'] = 'train_result/trocr-70M-en-zh-handwritten-small'
        CONFIG['num_train_epochs'] = 8
        CONFIG['per_device_train_batch_size'] = 32
        CONFIG['per_device_eval_batch_size'] = 64
        CONFIG['dataloader_num_workers'] = 8
    elif MODEL_SIZE == 'base':
        # 4card
        CONFIG['output_dir'] = 'train_result/trocr-70M-en-zh-handwritten-base'
        CONFIG['num_train_epochs'] = 4
        CONFIG['per_device_train_batch_size'] = 16
        CONFIG['per_device_eval_batch_size'] = 32
        CONFIG['dataloader_num_workers'] = 5

        # 8card
        # CONFIG['output_dir'] = 'train_result/trocr-70M-en-zh-handwritten-base'
        # CONFIG['num_train_epochs'] = 6
        # CONFIG['per_device_train_batch_size'] = 16
        # CONFIG['per_device_eval_batch_size'] = 32
        # CONFIG['dataloader_num_workers'] = 5
    elif MODEL_SIZE == 'large':
        # 8card
        CONFIG['output_dir'] = 'train_result/trocr-70M-en-zh-handwritten-large'
        CONFIG['num_train_epochs'] = 4
        CONFIG['per_device_train_batch_size'] = 11
        CONFIG['per_device_eval_batch_size'] = 22
        CONFIG['dataloader_num_workers'] = 3
    else:
        assert False

    training_args = TrainingArguments(**CONFIG)
    trainer = Trainer(
        model,
        training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        tokenizer=tokenizer, 
        data_collator=collate_fn_with_tokenizer,
    )
    trainer.train(resume_from_checkpoint=max_checkpoint_dir)
    # from scratch
    # trainer.train(resume_from_checkpoint=None)


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

    # MODEL_SIZE: Optional['small', 'base', 'large']
    MODEL_SIZE = os.getenv("MODEL_SIZE")

    # CACHE_DIR = '/groups/docintelli/home/u2023140841/huggigngface_cache_3years'
    EN_CACHE_DIR = '/groups/docintelli/home/u2023140841/huggigngface_cache_7years_en'
    ZH_CACHE_DIR = '/groups/docintelli/home/u2023140841/huggigngface_cache_7years_zh'
    HANDWRITTEN_ONLINE_CACHE_DIR = '/groups/docintelli/home/u2023140841/huggigngface_cache_7years_handwritten_online'
    HANDWRITTEN_NATURE_CACHE_DIR = '/groups/docintelli/home/u2023140841/huggigngface_cache_7years_handwritten_nature'
    en_dataset = load_dataset(
        '/groups/docintelli/home/u2023140841/code/TexTeller/src/models/ocr_model/train/data/loader.py',
        "en_formulas",
        num_proc=64,
        cache_dir=EN_CACHE_DIR,
    )['train']
    # zh_dataset = load_dataset(
    #     '/groups/docintelli/home/u2023140841/code/TexTeller/src/models/ocr_model/train/data/loader.py',
    #     "zh_formulas",
    #     num_proc=64,
    #     cache_dir=ZH_CACHE_DIR,
    # )['train']
    # handwritten_online_dataset = load_dataset(
    #     '/groups/docintelli/home/u2023140841/code/texteller/src/models/ocr_model/train/data/loader.py',
    #     "handwritten_online",
    #     num_proc=64,
    #     cache_dir=HANDWRITTEN_ONLINE_CACHE_DIR,
    # )['train']
    # handwritten_nature_dataset = load_dataset(
    #     '/groups/docintelli/home/u2023140841/code/texteller/src/models/ocr_model/train/data/loader.py',
    #     "handwritten_nature",
    #     num_proc=64,
    #     cache_dir=HANDWRITTEN_NATURE_CACHE_DIR,
    # )['train']
    # 扩大handwritten数据集的比例
    # ...
    # dataset = dataset.concatenate_datasets([en_dataset, zh_dataset, handwritten_online_dataset, handwritten_nature_dataset])
    dataset = en_dataset

    # tokenizer = TexTeller.get_tokenizer('/groups/docintelli/home/u2023140841/code/TexTeller/src/models/tokenizer/roberta-tokenizer-7Mformulas')
    tokenizer = TexTeller.get_tokenizer('/groups/docintelli/home/u2023140841/code/TexTeller/src/models/tokenizer/roberta-tokenizer-70M-en-zh')

    filter_fn_with_tokenizer = partial(filter_fn, tokenizer=tokenizer)
    dataset = dataset.filter(
        filter_fn_with_tokenizer,
        num_proc=128,
    )

    dataset = dataset.shuffle(seed=42)

    dataset = dataset.flatten_indices()

    map_fn = partial(tokenize_fn, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(
        map_fn, batched=True, remove_columns=dataset.column_names,
        num_proc=128,
        load_from_cache_file=True
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.0005, seed=42)
    train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']

    train_dataset = train_dataset.with_transform(img_train_transform)
    eval_dataset  = eval_dataset.with_transform(img_inf_transform)

    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    ckpt_path = Path(f'/groups/docintelli/home/u2023140841/code/TexTeller/src/models/ocr_model/train/train_result/trocr-70M-en-zh-handwritten-{MODEL_SIZE}')
    checkpoints = [
        int(dir.name.split('-')[1]) for dir in ckpt_path.iterdir() 
        if dir.is_dir() and dir.name.startswith('checkpoint-')
    ]
    max_checkpoint = max(checkpoints, default=None)
    max_checkpoint_dir = ckpt_path / f'checkpoint-{max_checkpoint}'
    print(f"ckpt_dir: {str(max_checkpoint_dir)}")

    model = TexTeller.from_pretrained(max_checkpoint_dir)
    # from scratch
    # model = TexTeller(size=MODEL_SIZE)

    # =================  debug  =======================
    # foo = train_dataset[:50]
    # bar = eval_dataset[:50]
    # =================  debug  =======================

    enable_train    = True
    enable_evaluate = True
    if enable_train:
        train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer)
    if enable_evaluate:
        evaluate(model, tokenizer, eval_dataset, collate_fn_with_tokenizer)

    os.chdir(cur_path)
