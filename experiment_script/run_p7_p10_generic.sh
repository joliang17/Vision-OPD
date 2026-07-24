#!/bin/bash
# P7 + P10 消融（2B × virl39k 90步，4+4卡并行）——通用认领脚本
# 用法（认领机器上）: TRIAL=trial<你的ARNOLD_TRIAL_ID> nohup bash experiment_script/run_p7_p10_generic.sh > logs/p7p10_stdout_$TRIAL.log 2>&1 &
#   P7  = contrast-标准只改 ra_contrast_alpha=0.5（gate关）  GPU0-3   —— paper α/gating 小节唯一钥匙
#   P10 = contrast-标准 + ra_no_sample_gate=True             GPU4-7   —— w_t 分解中间点
# 与 forward 对照(70.68)同口径: Qwen3-VL-2B, bs32, len6144, filter_overlong, 90步
# 修正了 301783374 版 driver 的两个坑:
#   1) 启动前 preflight 检查 tensorboard（301783374 就死在这上面, step0 即挂）
#   2) liveness 用训练日志 mtime + main_ppo 进程, 不用 pgrep 实验名（刚启动时 cmdline 里没有实验名, 会误报）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
: "${TRIAL:?set TRIAL=trial<ARNOLD_TRIAL_ID>}"
LOG=logs/p7p10_driver_${TRIAL}.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet

# preflight: 训练栈依赖（缺一个都会 step0 即死）
for mod in tensorboard termcolor num2words; do
  python3 -c "import $mod" 2>/dev/null || { log "FATAL: python 栈缺 $mod — 先 pip install 再跑（301783374 的死因）"; exit 1; }
done

# 双跑保护: 若已有其它 trial 的同名实验目录且步数>0, 先确认再跑
for n in contrast-alpha05-nogate contrast-standard-nosamplegate; do
  for d in checkpoints/Vision-OPD-${n}-Qwen3-VL-2B-virl39k-90step-trial*; do
    [ -d "$d" ] || continue
    case "$d" in *"$TRIAL") continue;; esac
    s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -gt 0 ] && { log "FATAL: $d 已有进度(step $s), 疑似别机在跑, 去 queue.md 对齐后再来"; exit 1; }
  done
done

wait_train() { # $1=ckpt目录 $2=日志 $3=目标step
  local d=$1 f=$2 tgt=$3
  while true; do
    local s; s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge "$tgt" ] && return 0
    if [ -f "$f" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      [ "$age" -gt 900 ] && ! pgrep -f verl.trainer.main_ppo >/dev/null 2>&1 && { log "DEAD: $d at step ${s}"; return 1; }
    fi
    sleep 300
  done
}
merge_steps() {
  for st in 30 60 90; do
    local dd="$1/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
      bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "$(basename "$1") step${st} merged"
  done
}

P7_NAME=Vision-OPD-contrast-alpha05-nogate-Qwen3-VL-2B-virl39k-90step-$TRIAL
P10_NAME=Vision-OPD-contrast-standard-nosamplegate-Qwen3-VL-2B-virl39k-90step-$TRIAL
P7_LOG=logs/p7_alpha05_nogate_2b_${TRIAL}_$(date +%Y%m%d_%H%M%S).log
P10_LOG=logs/p10_nosamplegate_2b_${TRIAL}_$(date +%Y%m%d_%H%M%S).log
log "launching P7 (GPU0-3) + P10 (GPU4-7) as $TRIAL"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$P7_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.5 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P7_LOG" 2>&1 &
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$P10_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_no_sample_gate=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P10_LOG" 2>&1 &
sleep 60
wait_train "checkpoints/$P7_NAME" "$P7_LOG" 90 && { merge_steps "checkpoints/$P7_NAME"; log "P7 DONE + merged"; } || log "P7 FAILED — 看 $P7_LOG"
wait_train "checkpoints/$P10_NAME" "$P10_LOG" 90 && { merge_steps "checkpoints/$P10_NAME"; log "P10 DONE + merged"; } || log "P10 FAILED — 看 $P10_LOG"
log "=== p7/p10 driver finished ($TRIAL) ==="
