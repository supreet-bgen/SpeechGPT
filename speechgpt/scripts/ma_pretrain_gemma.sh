#!/bin/bash

# -----------------------------
# DDP args
# -----------------------------
NNODE=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
NPROC=$5
last_checkpoint=$6

METAROOT="google/gemma-3-1b-pt"   # stage1
DATAROOT="data/en_all_audio_text/stage1"
OUTROOT="output/en_all_audio_text/stage1"
CACHEROOT="${DATAROOT}/cache/"

mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/
mkdir -p ${CACHEROOT}/group/train/
mkdir -p ${CACHEROOT}/group/valid/

# -----------------------------
# GPU selection
# -----------------------------
# Use first $NPROC GPUs by default
# export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC-1)))

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "NNODES=$NNODE NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NPROC=$NPROC"

# -----------------------------
# Resume flags
# -----------------------------
RESUME_FLAG=""
RESUME_WANDB=""
if [ -n "$last_checkpoint" ]; then
  RESUME_FLAG="--resume_from_checkpoint ${last_checkpoint}"
  RESUME_WANDB="--resume must"
fi

# -----------------------------
# Training
# -----------------------------
torchrun \
  --nproc_per_node=$NPROC \
  -m src.train.ma_pretrain_gemma \
    --report_to wandb \
    --run_name "gemma3_stage1_text+speech" \
    --bf16 True \
    --block_size 1024 \
    --model_name_or_path "${METAROOT}" \
    --train_file ${DATAROOT}/train.txt \
    --validation_files ${DATAROOT}/dev.txt ${DATAROOT}/dev_audio.txt ${DATAROOT}/dev_text.txt \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --preprocessing_num_workers 128 \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --log_level debug \
    --logging_steps 1 \
    --cache_dir ${CACHEROOT} \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 16 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --overwrite_cache True \
    $RESUME_FLAG \
    $RESUME_WANDB



    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'Gemma3DecoderLayer'
