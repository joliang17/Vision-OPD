set -euo pipefail
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
export HF_HOME=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/huggingface
export MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
export MODEL_SIZE=4B
export TRAINER_N_GPUS_PER_NODE=8
TRIAL_ID="${ARNOLD_TRIAL_ID}"

VIRL39K=data/virl39k_train_noimg_filtered_1img.parquet
SR1=data/vision_sr1_47k_noimg_v2_filtered.parquet

merge_prune_keep() {
  # merge_prune_keep <ckpt_dir> <step1> [<step2> ...]   -- keeps only the listed steps
  local ckpt_dir="$1"; shift
  local final; final="$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt")"
  bash scripts/merge_checkpoint.sh "${ckpt_dir}/global_step_${final}"
  for d in "${ckpt_dir}"/global_step_*; do
    step="$(basename "$d" | sed 's/global_step_//')"
    keepit=0
    for k in "$@"; do [[ "$step" == "$k" ]] && keepit=1; done
    [[ "$keepit" == "0" ]] && rm -rf "$d"
  done
}

########################################################################
# 2) visionopd对照的 GRPO baseline —— 默认数据(train_answer.parquet)，8卡
########################################################################
EXPERIMENT_NAME="Vision-OPD-grpo-baseline-default-trial${TRIAL_ID}" \
  bash scripts/run_experiment_grpo_baseline.sh trainer.max_actor_ckpt_to_keep=null
merge_prune_keep "checkpoints/Vision-OPD-grpo-baseline-default-trial${TRIAL_ID}" \
  "$(cat "checkpoints/Vision-OPD-grpo-baseline-default-trial${TRIAL_ID}/latest_checkpointed_iteration.txt")"
