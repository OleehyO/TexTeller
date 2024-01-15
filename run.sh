#!/usr/bin/env bash

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1,2,4

# 运行 Python 脚本并将输出重定向到日志文件
nohup python -m src.models.resizer.train.train > train_result_pred_height_v3.log 2>&1 &
