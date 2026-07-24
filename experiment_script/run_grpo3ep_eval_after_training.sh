#!/bin/bash
# 等 grpo-virl39k-3ep 训练完成后：merge 最终checkpoint + GPU7 跑 9-benchmark(server) + ZoomBench canonical
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

NAME="Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered-3ep"

echo "[$(date)] 等待 grpo-virl39k-3ep 训练进程结束..."
while pgrep -f "trainer.experiment_name=${NAME}" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] grpo-3ep 已结束"

FINAL_STEP=$(cat "checkpoints/${NAME}/latest_checkpointed_iteration.txt")
CKPT="checkpoints/${NAME}/global_step_${FINAL_STEP}"
if [ ! -d "${CKPT}/actor" ]; then
  echo "[$(date)] ERROR: ${CKPT}/actor 不存在，中止" >&2
  exit 1
fi

if [ ! -f "${CKPT}/config.json" ]; then
  bash scripts/merge_checkpoint.sh "${CKPT}" > "logs/merge_grpo3ep_step${FINAL_STEP}_$(date +%Y%m%d_%H%M%S).log" 2>&1
fi
echo "[$(date)] merge 完成 (step ${FINAL_STEP})，开始评测 (GPU7)"

cd VLMEvalKit
MODEL_PATH="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/${CKPT}" \
MODEL_NAME=grpo_virl39k_3ep_step${FINAL_STEP} \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=7 \
bash shell_scripts/eval_via_vllm_server.sh \
  > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_grpo3ep_step${FINAL_STEP}_301783374_$(date +%Y%m%d_%H%M%S).log 2>&1

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
bash scripts/run_zoombench_canonical.sh \
  "${CKPT}" \
  grpo_virl39k_3ep_step${FINAL_STEP} 7 8012 \
  > logs/zoombench_grpo3ep_step${FINAL_STEP}_301783374_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "[$(date)] grpo-3ep 最终checkpoint(step ${FINAL_STEP}) 全部评测完成"
