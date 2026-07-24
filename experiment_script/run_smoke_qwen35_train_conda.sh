#!/bin/bash
# 冒烟：验证本机(301783374)能否用 conda qwen35 env 训 Qwen3.5-4B（系统env没有qwen3_5架构）
# contrast-保守×virl39k 2步, GPU0,4,5,7, batch16。通过则由 run_qwen35_t3b_t6b_after_geo3k.sh 接正式任务
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
cd "$V"
export PATH="$CONDA_BIN:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
python3 -c "from transformers.models import qwen3_5; print('arch ok')" || exit 1

MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=1,2,3,6 TRAINER_N_GPUS_PER_NODE=4 \
EXPERIMENT_NAME=SMOKETEST-qwen35-cons-virl39k-trial301783374 \
ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
TRAIN_BATCH_SIZE=16 MAX_PROMPT_LENGTH=6144 \
bash scripts/run_experiment_contrast_conservative.sh \
data.filter_overlong_prompts=True trainer.total_training_steps=2 trainer.save_freq=2
echo "[$(date)] smoke exit=$?"
