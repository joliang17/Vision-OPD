set -uo pipefail
# mlx training job: ours (visionopd crop-teacher, no token weighting) on VisionOPD-6K, Qwen3.5-2B, 62 steps.
# Runs FOREGROUND so the mlx container stays alive until training + merge finish.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1"; export NO_PROXY="localhost,127.0.0.1,::1"

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"

MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
NAME=Vision-OPD-visionopd-Qwen3.5-4B-visionopd6k-62step-mlx
LOG=logs/mlxtrain_visionopd_qwen35_4b_vopd6k_$(date +%Y%m%d_%H%M%S).log
echo "[$(date)] launching $NAME (8卡 4B, bs32, len4096, 62步, visionopd crop-teacher)" | tee -a "$LOG"

MODEL_PATH="$MODEL" MODEL_SIZE=4B \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
EXPERIMENT_NAME="$NAME" TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
TRAINER_SAVE_FREQ=62 \
bash scripts/run_experiment_visionopd.sh \
  data.filter_overlong_prompts=True \
  trainer.total_training_steps=62 \
  trainer.max_actor_ckpt_to_keep=1 \
  >> "$LOG" 2>&1
RC=$?
echo "[$(date)] training exited rc=$RC" | tee -a "$LOG"
[ "$RC" -ne 0 ] && exit "$RC"

# only the final ckpt (step 62)
dd="checkpoints/$NAME/global_step_62"
if [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ]; then
  bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && echo "[$(date)] merged step62" | tee -a "$LOG"
fi
echo "[$(date)] DONE $NAME" | tee -a "$LOG"
