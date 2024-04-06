#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR="/home/lhy/code/TexTeller/src/models/ocr_model/model/ckpt"
export TOKENIZER_DIR="/home/lhy/code/TexTeller/src/models/tokenizer/roberta-tokenizer-7Mformulas"
export USE_CUDA=True  # True or False (case-sensitive)
export NUM_BEAM=3

streamlit run web.py
