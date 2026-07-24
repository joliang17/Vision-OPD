#!/bin/bash
# 新实验（2026-07-14 导师建议）：contrast-标准 + ctrl="真实图像 + 无关问题文本"
# ctrl 从 black(黑图,测图像依赖) 换成 qvis(保留图像、问题换成 "What is the answer?",测问题依赖)
# 机制:log p_hi − log p_ctrl 现在度量"这个 token 有多依赖看到真实问题",而不是"有多依赖看到图像"
# 对照组:contrast-标准(black ctrl) 7-bench 70.65 / 历史 qvis(旧权重机制,默认描述文本) 7-bench 67.20
# 等 GPU0-3 的既有任务(保守×sr1训练+ZoomBench补跑)全部退出后启动
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

echo "[$(date)] 等待 GPU0-3 的既有任务退出..."
while pgrep -f "run_qwen3vl_cleanup_after_grpo3ep.sh" >/dev/null 2>&1 || pgrep -f "run_cons_sr1_zoombench_after_cleanup.sh" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date)] GPU0-3 已空，启动 contrast-standard-qtext 训练"

CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT=qvis \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-qtext-Qwen3-VL-2B-Instruct \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  'actor_rollout_ref.actor.self_distillation.ra_generic_prompt=What is the answer?' \
  > logs/contrast_standard_qtext_2b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!
wait $PID
echo "[$(date)] qtext 训练结束"

NAME="Vision-OPD-contrast-standard-qtext-Qwen3-VL-2B-Instruct"
FINAL_STEP=$(cat "checkpoints/${NAME}/latest_checkpointed_iteration.txt" 2>/dev/null)
if [ -z "$FINAL_STEP" ]; then
  echo "[$(date)] ERROR: 没有checkpoint，训练失败" >&2
  exit 1
fi
CKPT="checkpoints/${NAME}/global_step_${FINAL_STEP}"
bash scripts/merge_checkpoint.sh "${CKPT}" > "logs/merge_qtext_step${FINAL_STEP}_$(date +%Y%m%d_%H%M%S).log" 2>&1

cd VLMEvalKit
BACKEND=vllm_server \
MODEL_PATH="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/${CKPT}" \
MODEL_NAME=contrast_standard_qtext_step${FINAL_STEP} \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_qtext_step${FINAL_STEP}_$(date +%Y%m%d_%H%M%S).log 2>&1

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
bash scripts/run_zoombench_canonical.sh "${CKPT}" contrast_standard_qtext_step${FINAL_STEP} 0 8030 \
  > logs/zoombench_qtext_step${FINAL_STEP}_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "[$(date)] qtext 实验全部完成(训练+merge+9bench+ZoomBench)"
