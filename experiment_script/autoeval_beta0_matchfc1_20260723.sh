#!/bin/bash
# 自动流水线:盯 β=0-matched-FC1 训练,checkpoint(30/60/90/120/150)一存好就自动 merge + 7-bench eval。
# eval 走原始栈(和 FC1 同口径可叠图),只用非训练卡(从 {3,4,5,6} 挑空闲的),不碰训练的 0/1/2/7。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/autoeval_beta0_matchfc1.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
NAME=Vision-OPD-contrast-uniform-beta0-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-matchfc1
CK=checkpoints/$NAME
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

# 从 {3,4,5,6} 挑一张空闲卡(mem<10GB),没有就等
pick_gpu(){
  while true; do
    for g in 3 4 5 6; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
      [ -n "$m" ] && [ "$m" -lt 10000 ] && { echo $g; return; }
    done
    sleep 30
  done
}

PORT=11201
for st in 30 60 90 120 150; do
  d=$CK/global_step_$st
  # 等训练越过该 step(latest>=st 保证 st 已完整写盘)
  log "等 checkpoint step$st ..."
  while [ "$(cat $CK/latest_checkpointed_iteration.txt 2>/dev/null||echo 0)" -lt "$st" ]; do sleep 30; done
  [ -d "$d/actor" ] || { log "WARN step$st 无 actor 分片,跳过"; continue; }
  # merge(前台,conda;已成功验证)
  if [ ! -f "$d/config.json" ]; then
    log "merge step$st ..."
    python3 -m verl.model_merger merge --backend fsdp --local_dir "$d/actor" --target_dir "$d" >> "$LOG" 2>&1
    [ -f "$d/config.json" ] || { log "ERROR step$st merge 失败,跳过"; continue; }
  fi
  # 挑卡 + eval(原始栈,复用通用脚本)
  g=$(pick_gpu); PORT=$((PORT+1))
  MN=beta0_matchfc1_step${st}
  log "eval step$st on GPU$g port$PORT (name=$MN)"
  bash experiment_script/eval_beta0_generic_7bench_20260723.sh "$d" "$MN" "$g" "$PORT" >> logs/autoeval_beta0_step${st}.log 2>&1
  sc=$(ls /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit/outputs_vllm_curated/${MN}_qwen3vl2b_temp0_4096_generic_eval/normal_scoring/*acc.csv 2>/dev/null|head -1 || true)
  log "step$st eval 完成 (acc.csv: ${sc:-检查中})"
done
log "=== β=0-matched-FC1 全部 5 个 checkpoint 自动 eval 完成 ==="
