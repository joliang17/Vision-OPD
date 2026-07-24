#!/bin/bash
# 等 cleanup 脚本(保守×sr1训练+9bench评测)退出后，补跑 cons_sr1 step30/60/90 的 ZoomBench canonical (GPU0)
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
NAME=checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step

while pgrep -f "run_qwen3vl_cleanup_after_grpo3ep.sh" >/dev/null 2>&1; do sleep 60; done
echo "[$(date)] cleanup 已退出，开始 cons_sr1 ZoomBench"

PORT=8027
for step in 30 60 90; do
  CKPT="${NAME}/global_step_${step}"
  if [ -f "${CKPT}/config.json" ]; then
    bash scripts/run_zoombench_canonical.sh "${CKPT}" cons_sr1_90step_step${step} 0 ${PORT} \
      > logs/backfill_zoombench_cons_sr1_step${step}_$(date +%Y%m%d_%H%M%S).log 2>&1
  else
    echo "[$(date)] SKIP cons_sr1 step${step}: 未merge(训练可能失败)" >> logs/task5_backfill_skips.log
  fi
  PORT=$((PORT+1))
done
echo "[$(date)] cons_sr1 ZoomBench 全部完成"
