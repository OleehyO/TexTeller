#!/usr/bin/env bash
set -exu

export CHECKPOINT_DIR="default"
export TOKENIZER_DIR="default"

streamlit run web.py
