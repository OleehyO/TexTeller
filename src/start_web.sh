#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR="OleehyO/TexTeller"
export TOKENIZER_DIR="OleehyO/TexTeller"
export USE_CUDA=False  # True or False (case-sensitive)
export NUM_BEAM=1

streamlit run web.py
