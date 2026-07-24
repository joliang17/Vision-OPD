#!/bin/bash
# visionopd repo-6k × ours,62步(=1 epoch),等 FCE + 本机两个 eval 全部跑完腾出 8 卡后自动启动
# 配置依据 queue.md 两台机器踩过的坑：
#   ① 必须 ra_uniform_weight=True（"ours"主配置惯例）
#   ② repo-6k 用默认 bs96 + total_epochs=1 跑满 ~62 步（5985/96≈62），不是 virl39k 的 bs32/90步截断
#   ③ 训练走 conda qwen35（项目标准）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/repo6k_ours_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
all_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

log "armed: 等 FCE + 本机 eval 全部结束(8卡全空)"
until all_free; do sleep 180; done
sleep 60; until all_free; do sleep 60; done   # 二次确认，防抖动
log "8卡全空，启动 repo-6k ours"

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

EXP=Vision-OPD-contrast-uniform-Qwen3-VL-2B-repo6k-1ep-trial301967423
CKPT=checkpoints/$EXP
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$EXP \
  ANSWER_VAL_TRAIN_FILE=$V/data/train_answer.parquet \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  > logs/repo6k_ours_$(date +%Y%m%d_%H%M%S).log 2>&1

l=$(step "$CKPT")
if [ "$l" -ge 60 ]; then
  for st in 20 40 62 $l; do
    d="$CKPT/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "step${st} merged"
  done
  log "repo-6k ours DONE (latest=$l) + merged"
else
  log "ERROR: repo-6k ours 只到 step $l"
fi
log "=== repo6k ours driver finished ==="
