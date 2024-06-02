#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
PEFT_PATH=$1

export CUDA_VISIBLE_DEVICES=0
python3 prompt_eval.py  \
    --peft_model_name_or_path_finetune ${PEFT_PATH} \
    