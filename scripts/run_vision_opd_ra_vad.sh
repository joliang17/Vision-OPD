#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_NAME="vopd"
DEFAULT_MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
if [[ -z "${MODEL_PATH:-}" ]]; then
    HF_CACHE_MODEL_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
    if [[ -d "$HF_CACHE_MODEL_DIR" ]]; then
        MODEL_PATH="$(find "$HF_CACHE_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    else
        MODEL_PATH="$DEFAULT_MODEL_PATH"
    fi
fi

EXPERIMENT="${EXPERIMENT:-black}"
TEACHER_MODEL_SOURCE="legacy"
TEACHER_REGULARIZATION="ema"
TEACHER_UPDATE_RATE=0.05

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-96}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
LR="${LR:-2e-6}"
DONT_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
ACTOR_USE_DYNAMIC_BSZ=True
PPO_MAX_TOKEN_LEN_PER_GPU=$TRAIN_MAX_MODEL_LEN
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
DETECTED_GPUS="$(python3 - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-$DETECTED_GPUS}"
TRAINER_NNODES="${WORLD_SIZE:-1}"
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:--1}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-null}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"tensorboard\"]}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-8}"
CUSTOM_CHAT_TEMPLATE_FILE="${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja"

DATA_DIR="${PROJECT_ROOT}/data"
if [[ -z "${TASK_TRAIN_FILE:-}" ]]; then
    if [[ "$EXPERIMENT" == "degrade" ]]; then
        TASK_TRAIN_FILE="${DATA_DIR}/train_degraded.parquet"
    else
        TASK_TRAIN_FILE="${DATA_DIR}/train.parquet"
    fi
fi

# degrade variant requires the offline-degraded parquet (run scripts/prepare_degraded_images.py first)
if [[ "$EXPERIMENT" == "degrade" && ! -f "$TASK_TRAIN_FILE" ]]; then
    echo "ERROR: degrade experiment needs $TASK_TRAIN_FILE." >&2
    echo "       Generate it first:  python3 scripts/prepare_degraded_images.py --input data/train.parquet --output data/train_degraded.parquet" >&2
    exit 1
fi

MODEL_NAME=$(basename "$DEFAULT_MODEL_PATH")
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-${EXPERIMENT}-${MODEL_NAME}}"
PROJECT_NAME="${PROJECT_NAME:-Vision-OPD}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
TRAINER_ROLLOUT_DATA_DIR="${TRAINER_ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}}"
mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"

EXTRA_ARGS=("$@")
# RA-VAD variants require full-vocab (non-top-k) distillation so student/teacher
# emit `all_logps` for the full-vocab KL. Set explicitly (do not rely on the default).
RA_FULL_LOGIT_ARG="actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
EXPERIMENT_ARGS=()
case "$EXPERIMENT" in
    visionopd)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.ra_vad=False
            actor_rollout_ref.actor.self_distillation.distillation_topk=100
            actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
            actor_rollout_ref.actor.self_distillation.is_clip=2.0
        )
        ;;
    baseline)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=answer_hint
            actor_rollout_ref.actor.self_distillation.ra_vad=False
            actor_rollout_ref.actor.self_distillation.distillation_topk=100
            actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
            actor_rollout_ref.actor.self_distillation.is_clip=2.0
        )
        ;;
    degrade)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.ra_vad=True
            actor_rollout_ref.actor.self_distillation.ra_ctrl_mode=degrade
            actor_rollout_ref.actor.self_distillation.ra_ctrl_image_key=images_degraded
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$RA_FULL_LOGIT_ARG"
        )
        ;;
    qvis)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.ra_vad=True
            actor_rollout_ref.actor.self_distillation.ra_ctrl_mode=qvis
            actor_rollout_ref.actor.self_distillation.ra_ctrl_image_key=images
            actor_rollout_ref.actor.self_distillation.ra_generic_prompt="Describe this image in detail."
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$RA_FULL_LOGIT_ARG"
        )
        ;;
    noimg)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.ra_vad=True
            actor_rollout_ref.actor.self_distillation.ra_ctrl_mode=noimg
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$RA_FULL_LOGIT_ARG"
        )
        ;;
    black)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.ra_vad=True
            actor_rollout_ref.actor.self_distillation.ra_ctrl_mode=black
            actor_rollout_ref.actor.self_distillation.ra_ctrl_image_key=images
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$RA_FULL_LOGIT_ARG"
        )
        ;;
    *)
        echo "Unknown EXPERIMENT=$EXPERIMENT. Use visionopd|baseline|degrade|qvis|noimg|black." >&2
        exit 1
        ;;
esac

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

# --- Offline / no-tracking safety --------------------------------------------
# Force offline mode so no HF / dataset downloads are attempted over the network.
# The model must already be in the HF cache (MODEL_PATH resolves to the local
# snapshot above) and training data is a local parquet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# No wandb: stick to console + tensorboard (TRAINER_LOGGER default already excludes wandb).
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
    if [[ ! -f "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
        echo "Custom chat template file not found: ${CUSTOM_CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_ARGS+=(actor_rollout_ref.model.custom_chat_template_file="$CUSTOM_CHAT_TEMPLATE_FILE")
fi

echo "Running: $EXPERIMENT_NAME"
echo "Experiment: $EXPERIMENT"
echo "Train file: $TASK_TRAIN_FILE"

python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MIMI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.ra_temperature=2.0 \
    actor_rollout_ref.actor.self_distillation.ra_delta=0.0 \
    actor_rollout_ref.actor.self_distillation.ra_clip_quantile=0.95 \
    actor_rollout_ref.actor.self_distillation.ra_min_positive_tokens=1 \
    actor_rollout_ref.actor.self_distillation.ra_no_sample_gate=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${EXPERIMENT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
