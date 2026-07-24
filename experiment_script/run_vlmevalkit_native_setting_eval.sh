#!/bin/bash
# 【顺序1/2，用户要求先跑这个】用 VisionOPD 原生eval的采样配置(max_tokens=8192, presence_penalty=0)
# 在 VLMEvalKit server模式跑同配置eval。
# 目的：区分"原生 vs VLMEvalKit 分数差异"里有多少来自 max_new_tokens=4096 太短 / presence_penalty=1.5
# 同 checkpoint(contrast-标准-2B step62)、同3个benchmark(VStarBench/HRBench4K/HRBench8K)，GPU4
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

echo "[$(date)] 等待 grpo-virl39k-3ep 训练进程结束..."
while pgrep -f "trainer.experiment_name=Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered-3ep" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] grpo-3ep 已结束，启动原生配置版 VLMEvalKit eval (GPU4)"

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
BACKEND=vllm_server \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct/global_step_62 \
MODEL_NAME=native_setting_std_step62_8192_pp0 \
DATASETS=VStarBench,HRBench4K,HRBench8K \
MAX_NEW_TOKENS=8192 \
PRESENCE_PENALTY=0 \
GPU_IDS=4 \
nohup bash shell_scripts/eval_model_temp0_4096.sh \
  > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/vlmevalkit_native_setting_eval_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] 原生配置版 VLMEvalKit eval 完成"
