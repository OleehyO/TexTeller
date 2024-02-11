import evaluate
import numpy as np
import os

from pathlib import Path
from typing import Dict
from transformers import EvalPrediction, RobertaTokenizer


def bleu_metric(eval_preds: EvalPrediction, tokenizer: RobertaTokenizer) -> Dict:
    cur_dir = Path(os.getcwd())
    os.chdir(Path(__file__).resolve().parent)
    metric = evaluate.load('google_bleu')  # Will download the metric from huggingface if not already downloaded
    os.chdir(cur_dir)
    
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    preds = logits

    labels = np.where(labels == -100, 1, labels)

    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=preds, references=labels)
