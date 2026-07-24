#!/bin/bash
# 等本机 grpo-virl39k-3ep 跑完后，收尾：保守×sr1-90step 从头训练(4096, GPU0-3)+merge+评测30/60/90
# (原来的标准×virl39k step30 部分已挪到 run_std_virl39k_3060_full_eval.sh，避免重复)
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

echo "[$(date)] 等待 grpo-virl39k-3ep 训练进程结束..."
while pgrep -f "trainer.experiment_name=Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered-3ep" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] grpo-3ep 已结束"

# --- 保守×sr1-90step 从头训练(4096) ---
CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_conservative_sr1_90step_301783374_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!
wait $PID
echo "[$(date)] 保守×sr1-90step 训练结束"

# --- merge + 评测 step30/60/90 ---
NAME="Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step"
for step in 30 60 90; do
  ckpt_dir="checkpoints/${NAME}/global_step_${step}"
  if [ -d "${ckpt_dir}/actor" ] && [ ! -f "${ckpt_dir}/config.json" ]; then
    bash scripts/merge_checkpoint.sh "${ckpt_dir}" > "logs/merge_cons_sr1_301783374_step${step}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  fi
done

cd VLMEvalKit
for step in 30 60 90; do
  ckpt_dir="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/${NAME}/global_step_${step}"
  if [ -f "${ckpt_dir}/config.json" ]; then
    BACKEND=vllm_server \
    MODEL_PATH="${ckpt_dir}" \
    MODEL_NAME="task5_cons_sr1_step${step}_server" \
    DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
    GPU_IDS=0 \
    nohup bash shell_scripts/eval_model_temp0_4096.sh \
      > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_task5_cons_sr1_step${step}_301783374_$(date +%Y%m%d_%H%M%S).log 2>&1
  fi
done
echo "[$(date)] 保守×sr1-90step 全部merge+评测完成"
