#!/bin/bash

# DeepSpeed Team

CURRENT_TIME=$(TZ=Asia/Seoul date +"%Y-%m-%d-%H.%M.%S")

ZERO_STAGE="--zero_stage 2"

MODEL_NAME=$1
MODEL_PATH=$2
OUTPUT='/home/chanho/Model/SHARE/Refactorizing/result/output_path'

export TOKENIZERS_PARALLELISM=False

TRN_FN='/home/chanho/Model/SHARE/Refactorizing/result/dataset/train_without_tag.json'
DEV_FN='/home/chanho/Model/SHARE/Refactorizing/result/dataset/valid_without_tag.json'


TOTAL_SIZE=`wc -l ${TRN_FN}`
echo "number of samples in trainset: ${TOTAL_SIZE}"

mkdir -p $OUTPUT/$CURRENT_TIME
deepspeed --include localhost:0,1 \
--master_port 12341 \
training/EPISODE/main.py \
   --model_name ${MODEL_NAME} \
   --model_name_or_path ${MODEL_PATH} \
   --train_data_path ${TRN_FN} \
   --valid_data_path ${DEV_FN} \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --data_output_path $OUTPUT/data \
   --max_seq_len 2048 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 50 \
   --num_train_samples ${TOTAL_SIZE} \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 1000 \
   --seed 42 \
   ${ZERO_STAGE} \
   --save_interval 1000 \
   --eval_interval 100 \
   --output_dir $OUTPUT/$CURRENT_TIME \
 &>$OUTPUT/$CURRENT_TIME/train.log&

   

      
   

