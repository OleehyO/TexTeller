@echo off
SETLOCAL ENABLEEXTENSIONS

set CHECKPOINT_DIR=default
set TOKENIZER_DIR=default

streamlit run web.py

ENDLOCAL
