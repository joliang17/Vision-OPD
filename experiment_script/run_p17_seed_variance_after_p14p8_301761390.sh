#!/bin/bash
# trial 301761390: P17 seed-variance 复跑（2026-07-17 认领）
# contrast-标准 × virl39k-filtered 2B 90步，与主表 ours 行（70.68）唯一差异 = data.seed=1234
# seed 布线验证结论（07-17，本 session 代码追踪；Codex 复核待其重新登录）：
#   - data.seed → main_ppo.py:470-475 torch.Generator().manual_seed(seed) → RandomSampler → 训练数据顺序 ✅ 生效
#   - data.seed=null（历史所有 run）时 torch.Generator() 默认 initial_seed=67280421310721 固定值
#     ⇒ 历史 run 数据顺序全部相同；本实验是第一个真正换数据顺序的 run
#   - rollout vLLM 引擎 seed: RolloutConfig dataclass 无 seed 字段, vllm_async_server.py:320
#     config.get("seed",0) 恒为 0, 不改代码无法覆盖——无碍: 数据顺序一变, 权重轨迹从 step1 分叉,
#     rollout 内容自然全不同
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p17_seed_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet
NAME=Vision-OPD-contrast-standard-seed1234-Qwen3-VL-2B-virl39k-90step-trial301761390

log "=== waiting for P14/P8 driver to finish ==="
P148_PID=$(ps aux | grep -v grep | grep run_p14_p8_after_p6_301761390.sh | awk '{print $2}' | head -1)
log "p14/p8 driver pid=${P148_PID:-unknown}"
while true; do
  grep -q "P14/P8 driver finished" logs/p14_p8_driver_301761390.log && break
  if [ -n "${P148_PID:-}" ]; then
    kill -0 "$P148_PID" 2>/dev/null || { log "WARN: p14/p8 driver 死了但没写结束标记, 请人工核对"; break; }
  fi
  sleep 300
done
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done

TLOG=logs/p17_seed1234_$(date +%Y%m%d_%H%M%S).log
log "launching P17 seed-variance (8 GPUs, data.seed=1234)"
MODEL_SIZE=2B TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  data.seed=1234 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$TLOG" 2>&1 &
sleep 60
while true; do
  s=$(cat "checkpoints/$NAME/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  [ "${s:-0}" -ge 90 ] && break
  age=$(( $(date +%s) - $(stat -c %Y "$TLOG" 2>/dev/null || date +%s) ))
  [ "$age" -gt 900 ] && ! ps aux | grep -v grep | grep -q verl.trainer.main_ppo && { log "DEAD at step ${s} — 看 $TLOG"; exit 1; }
  sleep 300
done
for st in 30 60 90; do
  dd="checkpoints/$NAME/global_step_${st}"
  [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
    bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "step${st} merged"
done
log "=== P17 DONE + merged ==="
