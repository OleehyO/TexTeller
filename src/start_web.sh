#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR="/home/lhy/code/TexTeller/src/models/ocr_model/train/train_result/TexTellerv3/checkpoint-788000"
export TOKENIZER_DIR="default"

streamlit run web.py
