#!/bin/bash
# 验证"prompt风格=5pp根因"：VLMEvalKit + 官方采样配置 + VLMEVAL_MCQ_OFFICIAL_STYLE=1(官方prompt风格)
# 如果 HRBench4K 从 76 档回升到 81 档 → 实锤并给出 VLMEvalKit 里复刻官方数字的完整配方
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit

VOPD=/mnt/bn/tns-algo-video-vlm-ruby/yijiangli/project/opsd/Vision-OPD/checkpoints/Vision-OPD-Qwen3-VL-4B-Instruct/global_step_65

BACKEND=vllm_server PORT=28805 \
VLMEVAL_MCQ_OFFICIAL_STYLE=1 \
TEMPERATURE=0.7 TOP_P=0.8 TOP_K=20 PRESENCE_PENALTY=1.5 MAX_NEW_TOKENS=8192 \
MODEL_PATH="$VOPD" MODEL_NAME=visionopd4b_officialprompt_vlmevalkit \
DATASETS=VStarBench,HRBench4K,HRBench8K GPU_IDS=6 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > "$V/logs/officialprompt_vlmevalkit_visionopd4b_$(date +%Y%m%d_%H%M%S).log" 2>&1
echo "[$(date)] 官方prompt风格 VLMEvalKit 验证完成"
