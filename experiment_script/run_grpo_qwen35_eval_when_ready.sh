#!/bin/bash
# 等 GRPO×Qwen3.5(本仓库数据, 301761390在训) 训完+merge后自动评测(9-bench+ZoomBench, conda shim)
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
SHIM=$V/scripts/qwen35_shim
CKPT_DIR=$V/checkpoints/Vision-OPD-grpo-baseline-default-Qwen3.5-4B-trial301761390
DS="BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench"

echo "[$(date)] 等 GRPO×Qwen3.5 训练完成+merge(以最新step目录出现config.json为信号)..."
while true; do
  FS=$(cat $CKPT_DIR/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
  [ "$FS" -gt 0 ] && [ -f "$CKPT_DIR/global_step_$FS/config.json" ] && break
  sleep 300
done
echo "[$(date)] GRPO×Qwen3.5 ready(step $FS)，等空闲GPU3..."
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==4{exit !($1>5000)}'; do sleep 120; done

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
PATH=$SHIM:$PATH BACKEND=vllm_server PORT=28841 \
MODEL_PATH=$CKPT_DIR/global_step_$FS \
MODEL_NAME=grpo_qwen35_4b_step$FS DATASETS=$DS GPU_IDS=3 \
bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_grpo_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
cd $V
PATH=$SHIM:$PATH bash scripts/run_zoombench_canonical.sh \
  $CKPT_DIR/global_step_$FS grpo_qwen35_4b_step$FS 3 28842 > logs/zoombench_grpo_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] GRPO×Qwen3.5 评测完成"
