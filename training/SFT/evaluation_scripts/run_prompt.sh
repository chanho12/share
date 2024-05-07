#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

FINETUED_PATH=$1
PEFT_PATH=$2


# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0,1
python3 prompt_eval.py \
    --model_name_or_path_finetune ${FINETUED_PATH} \
    --peft_model_name_or_path ${PEFT_PATH} \
