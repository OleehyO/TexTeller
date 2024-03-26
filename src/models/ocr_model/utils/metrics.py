import evaluate
import numpy as np
from transformers import EvalPrediction, RobertaTokenizer
from typing import Dict

def bleu_metric(eval_preds:EvalPrediction, tokenizer:RobertaTokenizer) -> Dict:
    metric = evaluate.load('/home/lhy/code/TexTeller/src/models/ocr_model/train/google_bleu')  # 这里需要联网，所以会卡住
    
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    preds = logits
    # preds = np.argmax(logits, axis=1)  # 把logits转成对应的预测标签

    labels = np.where(labels == -100, 1, labels)

    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=preds, references=labels)