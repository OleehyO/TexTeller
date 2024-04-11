@echo off
SETLOCAL ENABLEEXTENSIONS

set CHECKPOINT_DIR=default
set TOKENIZER_DIR=default
set USE_CUDA=False  REM True or False (case-sensitive)
set NUM_BEAM=1

streamlit run web.py

ENDLOCAL
