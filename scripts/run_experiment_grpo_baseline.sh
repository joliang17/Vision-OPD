#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-baseline_grpo}"

DEFAULT_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$DEFAULT_MODEL_PATH"
  CANDIDATE_SNAPSHOT_DIRS=(
    "$PROJECT_ROOT/../../cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"
    "$PROJECT_ROOT/../../cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"
    "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"
  )
  for snapshot_dir in "${CANDIDATE_SNAPSHOT_DIRS[@]}"; do
    [[ -d "$snapshot_dir" ]] || continue
    while IFS= read -r candidate; do
      if [[ -f "$candidate/model.safetensors" || -f "$candidate/model.safetensors.index.json" || -f "$candidate/pytorch_model.bin" ]]; then
        MODEL_PATH="$candidate"
      fi
    done < <(find "$snapshot_dir" -mindepth 1 -maxdepth 1 -type d | sort)
    [[ "$MODEL_PATH" != "$DEFAULT_MODEL_PATH" ]] && break
  done
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-grpo-baseline-Qwen3-VL-2B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-Vision-OPD}"
TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${PROJECT_ROOT}/data/train.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LR="${LR:-1e-6}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-$(python3 - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}"
TRAINER_NNODES="${WORLD_SIZE:-1}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-null}"
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-20}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:--1}"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-null}"

ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.0}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

LOCAL_DIR="${LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}}"
REWARD_FILE="${REWARD_FILE:-${PROJECT_ROOT}/verl/utils/reward_score/vision_opd_vqa.py}"
CUSTOM_CHAT_TEMPLATE_FILE="${CUSTOM_CHAT_TEMPLATE_FILE:-${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja}"
mkdir -p "$LOCAL_DIR" "$ROLLOUT_DIR"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

CHAT_TEMPLATE_ARGS=()
if [[ -n "$CUSTOM_CHAT_TEMPLATE_FILE" && -f "$CUSTOM_CHAT_TEMPLATE_FILE" ]]; then
  CHAT_TEMPLATE_ARGS+=(actor_rollout_ref.model.custom_chat_template_file="$CUSTOM_CHAT_TEMPLATE_FILE")
fi

echo "Running GRPO baseline: $EXPERIMENT_NAME"
echo "Model: $MODEL_PATH"
echo "Reward: $REWARD_FILE, format_weight=$FORMAT_REWARD_WEIGHT"

python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
  data.train_files="[\"$TASK_TRAIN_FILE\"]" \
  data.val_files="[]" \
  data.filter_overlong_prompts=False \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.truncation=error \
  data.shuffle=True \
  data.trust_remote_code=True \
  data.return_multi_modal_inputs=True \
  data.image_key=images \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  reward_model.use_reward_loop=False \
  custom_reward_function.path="$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.format_weight="$FORMAT_REWARD_WEIGHT" \
  critic.model.path="$MODEL_PATH" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.group_name="$EXPERIMENT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER" \
  trainer.n_gpus_per_node="$TRAINER_N_GPUS_PER_NODE" \
  trainer.nnodes="$TRAINER_NNODES" \
  trainer.save_freq="$TRAINER_SAVE_FREQ" \
  trainer.test_freq="$TRAINER_TEST_FREQ" \
  trainer.total_epochs="$TRAINER_TOTAL_EPOCHS" \
  trainer.total_training_steps="$TRAINER_TOTAL_TRAINING_STEPS" \
  trainer.max_actor_ckpt_to_keep="$TRAINER_MAX_ACTOR_CKPT_TO_KEEP" \
  trainer.val_before_train=False \
  trainer.default_local_dir="$LOCAL_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  "${CHAT_TEMPLATE_ARGS[@]}" \
  "$@"
