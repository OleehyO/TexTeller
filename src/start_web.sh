#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR="/home/lhy/code/TexTeller/src/models/ocr_model/train/train_result/TexTellerv3/checkpoint-460000"
# export CHECKPOINT_DIR="default"
export TOKENIZER_DIR="/home/lhy/code/TexTeller/src/models/tokenizer/roberta-tokenizer-7Mformulas"
export USE_CUDA=True  # True or False (case-sensitive)
export NUM_BEAM=3

streamlit run web.py
