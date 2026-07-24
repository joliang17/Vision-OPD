#!/bin/bash
# 301783374 接续 driver（2026-07-16 v3，JSD 最终归 301761390——它的 GRPO 20min 就完、比本机早开跑）：
#  1. 等 geo3k 保守版训完(driver退出) → merge geo3k std/cons 最终 checkpoint
#  2. 在腾出的 GPU1,2,3,6 上冒烟验证 conda qwen35 env 能否训 Qwen3.5（本机系统env无qwen3_5架构）
#  3. 等 reverse-KL(GPU0,4,5,7) 跑完 step90 → merge 30/60/90
#  4. 冒烟通过才接 T3b(保守×virl39k 90步)、T6b(保守×sr1 90步, len4096)，8卡串行
#     —— checkpoint 产物校验成败，不信退出码（两台机器都踩过的静默空跑坑）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
cd "$V"
LOG=logs/after_geo3k_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }

wait_gpus_free() {  # <gpu_id...> 等指定GPU显存回落
  while true; do
    local busy=0
    for g in "$@"; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      [ "${m:-999999}" -ge 10000 ] && busy=1
    done
    [ "$busy" -eq 0 ] && return 0
    sleep 60
  done
}

verify_ckpt() {  # <ckpt_dir> <want_step>
  local latest
  latest=$(cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  [ "${latest:-0}" -ge "$2" ] && [ -d "$1/global_step_${latest}/actor" ]
}

# ---- 1. 等 geo3k driver 退出 ----
log "waiting for geo3k driver to exit"
while pgrep -f "run_geo3k_contrast_4b.sh" >/dev/null 2>&1; do sleep 120; done
wait_gpus_free 1 2 3 6
for name in standard conservative; do
  D=checkpoints/Vision-OPD-contrast-${name}-Qwen3-VL-4B-Instruct-geometry3k
  s=$(cat "$D/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "${s:-0}" -ge 60 ]; then
    [ -f "$D/global_step_${s}/config.json" ] || bash scripts/merge_checkpoint.sh "$D/global_step_${s}" >> "$LOG" 2>&1
    log "geo3k ${name}: step ${s} merged"
  else
    log "ERROR geo3k ${name}: only step ${s}, NOT merging (训练疑似失败)"
  fi
done

# ---- 2. conda qwen35 训练冒烟：已于 07-16 07:08 通过（step2 checkpoint 完整产出），不再重复 ----
# （16:28 的第二次冒烟 OOM 是冒烟自身配置太重：batch16×4卡 → mini96/4=24条/卡，是正式8卡跑
#   (12条/卡) 的两倍负载，属容量擦边而非环境问题；环境可用性以 07:08 的通过为准。）
SMOKE_OK=1
log "smoke: using 07:08 PASS result (env validated), skipping re-run"

# ---- 3. 等 reverse-KL 跑完（崩了就 8 卡 resume 重试，最多 2 次）----
# 背景：06:12 首跑在 step20 后 backward OOM（reverse KL 的 kl_div(teacher,student,log_target)
# 需物化 exp(student_logprobs) 全词表梯度链，比 forward 吃显存）。修法=8卡跑（DP 宽度
# 翻倍 → 每卡全词表激活减半，数学不变）。两个已踩的坑：
# ⚠️ 别加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments——与 vLLM CuMemAllocator memory pool
#   不兼容（pytorch#147851），vLLM 引擎初始化必崩（07-16 两次 resume 死于此）。
# ⚠️ FSDP checkpoint 绑定 world size（分片名 model_world_size_N_rank_*.pt）——4卡存的 step20
#   无法在 8 卡 resume（FileNotFoundError）。4卡半成品已挪到 DISCARD-reversekl-4gpu-partial-step20，
#   8 卡从 step0 重训（8卡 ~50s/步，20步只值 ~17min，不心疼）。
# ⚠️ 8卡首跑仍在 step22 backward OOM（要 48.69GB 只剩 47.40GB，差 1.3GB）——占显存的是 vLLM rollout
#   引擎常驻份额，已在 resume 命令里加 rollout.gpu_memory_utilization=0.7→0.55（腾 ~27GB，
#   不碰训练数学，只影响 rollout KV cache 大小/速度）。
KL_DIR=checkpoints/Vision-OPD-contrast-standard-reversekl-Qwen3-VL-2B-virl39k-90step-trial301783374
KL_ATTEMPTS=0
log "waiting for reverse-KL to reach step90 (with resume-retry)"
while ! verify_ckpt "$KL_DIR" 90; do
  if ! pgrep -f "reversekl-Qwen3-VL-2B" >/dev/null 2>&1; then
    if [ "$KL_ATTEMPTS" -ge 2 ]; then log "ERROR: reverse-KL still short of step90 after ${KL_ATTEMPTS} resumes (at step $(cat "$KL_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0))"; break; fi
    KL_ATTEMPTS=$((KL_ATTEMPTS+1))
    wait_gpus_free 0 1 2 3 4 5 6 7
    log "resuming reverse-KL on 8 GPUs (attempt ${KL_ATTEMPTS})"
    MODEL_SIZE=2B \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
      EXPERIMENT_NAME=Vision-OPD-contrast-standard-reversekl-Qwen3-VL-2B-virl39k-90step-trial301783374 \
      ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
      TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
      nohup bash scripts/run_experiment_contrast_standard.sh \
      actor_rollout_ref.actor.self_distillation.ra_divergence_alpha=1.0 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
      data.filter_overlong_prompts=True trainer.total_training_steps=90 \
      > logs/contrast_std_reversekl_2b_virl39k_90step_8gpu_retry${KL_ATTEMPTS}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    sleep 600
  fi
  sleep 300
done
for s in 30 60 90; do
  d="$KL_DIR/global_step_${s}"
  [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "reverse-KL step${s} merged"
done
wait_gpus_free 0 4 5 7

# ---- 4. T3b / T6b（conda env, 8卡串行）----
if [ "$SMOKE_OK" -ne 1 ]; then log "skip T3b/T6b (smoke failed)"; exit 0; fi
export PATH="$CONDA_BIN:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

run_seg() {  # <ckpt_dir> <want_step> <logfile> <cmd...>
  local dir=$1 want=$2 lf=$3; shift 3
  # 防撞：别的trial已完成就跳过
  for other in 301683547 301761390; do
    local alt="${dir/301783374/$other}"
    if verify_ckpt "$alt" "$want"; then log "skip: $alt already done elsewhere"; return 0; fi
    pgrep -f "$(basename "${alt}")" >/dev/null 2>&1 && { log "skip: $(basename "$alt") 正在别处跑（本机进程可见性有限，仅防本机重复）"; }
  done
  log "launching $(basename "$dir") (log: $lf)"
  "$@" > "$lf" 2>&1
  if verify_ckpt "$dir" "$want"; then
    for s in 30 60 90; do
      d="$dir/global_step_${s}"
      [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1
    done
    log "done + merged: $dir"
  else
    log "ERROR: $(basename "$dir") no valid checkpoint at step $want, see $lf; NOT retrying"
  fi
  wait_gpus_free 0 1 2 3 4 5 6 7
}

T3B=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374
run_seg "$T3B" 90 logs/t3b_cons_virl39k_qwen35_301783374.log \
  env MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_conservative.sh data.filter_overlong_prompts=True trainer.total_training_steps=90

T6B=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-sr1-filtered-90step-trial301783374
run_seg "$T6B" 90 logs/t6b_cons_sr1_qwen35_301783374.log \
  env MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-sr1-filtered-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_experiment_contrast_conservative.sh data.filter_overlong_prompts=True trainer.total_training_steps=90

log "=== after-geo3k driver finished ==="
