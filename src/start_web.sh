#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR="default"
export TOKENIZER_DIR="default"
# export USE_CUDA=False  # True or False (case-sensitive)
export USE_CUDA=True  # True or False (case-sensitive)
export NUM_BEAM=10

streamlit run web.py
