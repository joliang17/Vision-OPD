#!/bin/bash
# 2026-07-22（用户下达）：P33(no-anchor/pure-contrast target) 换个 seed 重训，看训练是否稳定
#（P33 本身是高优先级的"必须跑"消融，之前 301832790 已用默认seed跑完；这次用 seed=1234，
#  是本项目已有惯例的"第二seed"值，跟S2b/QS1等其它seed对照点一致，方便复用比较基准）。
# 除 seed 外配置完全照抄 experiment_script/run_p33_pure_contrast_noanchor_2b.sh：
#   uniform weight, β=0.1(默认), EOS-exempt(默认), unfiltered virl39k, anchor_coef=0, alpha=1.0,
#   2B, len6144, bs32, 90步。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p33_noanchor_seed1234_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

wait_gpus_free() { while true; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "${m:-999999}" -lt 5000 ] && break; sleep 60; done; }
wait_train() {
  local d=$1 f=$2 tgt=$3
  while true; do
    local s; s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge "$tgt" ] && return 0
    if [ -f "$f" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      [ "$age" -gt 900 ] && ! ps aux | grep -v grep | grep -q verl.trainer.main_ppo && { log "DEAD: $d at step ${s}"; return 1; }
    fi
    sleep 300
  done
}

log "=== P33 no-anchor, seed=1234, 2B uniform x unfiltered, len6144, 90步 (稳定性对照) ==="
wait_gpus_free
NAME=Vision-OPD-pure-contrast-noanchor-uniform-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832756
rm -rf "checkpoints/$NAME"
RUN_LOG=logs/p33_noanchor_seed1234_$(date +%Y%m%d_%H%M%S).log
log "launching (8卡): $NAME"
EXPERIMENT_NAME=$NAME MODEL_SIZE=2B TRAINER_N_GPUS_PER_NODE=8 \
  ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=1.0 \
  actor_rollout_ref.actor.self_distillation.ra_contrast_anchor_coef=0.0 \
  data.seed=1234 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$RUN_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$NAME" "$RUN_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "step${st} merged"
  done
  log "P33 seed1234 DONE + merged"
else
  log "FAILED/CRASHED — 看 $RUN_LOG（这本身就是稳定性问题的信号，记下崩在哪一步）"
fi
log "=== driver finished ==="
