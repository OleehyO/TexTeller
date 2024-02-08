#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR=/home/lhy/code/TeXify/src/models/ocr_model/model_checkpoint
export TOKENIZER_DIR=/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550K
export USE_CUDA=False
export NUM_BEAM=3

streamlit run web.py
