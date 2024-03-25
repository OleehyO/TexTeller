#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR="default"
export TOKENIZER_DIR="default"
export USE_CUDA=False  # True or False (case-sensitive)
export NUM_BEAM=1

streamlit run web.py
