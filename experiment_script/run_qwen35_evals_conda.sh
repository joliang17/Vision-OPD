#!/bin/bash
# Qwen3.5-4B 欠账评测(用户提供 conda qwen35 环境; 已修复 hub/scipy/joblib 依赖):
#   GPU1: baseline(answer-hint)×本仓库 step62 — 9-bench
#   GPU2: baseline×virl39k step145 — 9-bench
#   GPU3: visionopd step62 HRBench8K 重跑(原73.00被56条API失败污染) → 2B answerhint 干净重跑(9-bench,系统vllm)
# 机制: PATH shim 让 `vllm` 用 conda 的(0.18,支持Qwen3.5), VLMEvalKit 客户端仍是 /usr/bin/python
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
SHIM=$V/scripts/qwen35_shim
DS="BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench"

echo "[$(date)] 等 geometry3k 冒烟(GPU1-3)退出..."
while pgrep -f "SMOKETEST-contrast-standard-Qwen3-VL-4B-geometry3k" >/dev/null 2>&1; do sleep 30; done

lane1() {
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  PATH=$SHIM:$PATH BACKEND=vllm_server PORT=28821 \
  MODEL_PATH=$V/checkpoints/Vision-OPD-baseline-Qwen3.5-4B-trial301761390/global_step_62 \
  MODEL_NAME=baseline_qwen35_4b_repo_step62 DATASETS=$DS GPU_IDS=1 \
  bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_baseline_qwen35_repo_$(date +%Y%m%d_%H%M%S).log 2>&1
  echo "[$(date)] lane1 done"
}
lane2() {
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  PATH=$SHIM:$PATH BACKEND=vllm_server PORT=28822 \
  MODEL_PATH=$V/checkpoints/Vision-OPD-baseline-Qwen3.5-4B-virl39k-filtered-trial301761390/global_step_145 \
  MODEL_NAME=baseline_qwen35_4b_virl39k_step145 DATASETS=$DS GPU_IDS=2 \
  bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_baseline_qwen35_virl39k_$(date +%Y%m%d_%H%M%S).log 2>&1
  echo "[$(date)] lane2 done"
}
lane3() {
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  PATH=$SHIM:$PATH BACKEND=vllm_server PORT=28823 \
  MODEL_PATH=$V/checkpoints/Vision-OPD-visionopd-Qwen3.5-4B-trial301761390/global_step_62 \
  MODEL_NAME=visionopd_qwen35_4b_hrbench8k_rerun DATASETS=HRBench8K GPU_IDS=3 \
  bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_visionopd_qwen35_hrbench8k_rerun_$(date +%Y%m%d_%H%M%S).log 2>&1
  # 2B answerhint 干净重跑(Qwen3-VL,系统vllm,无shim)
  BACKEND=vllm_server PORT=28824 \
  MODEL_PATH=$V/checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-Instruct-trial301683547/global_step_62 \
  MODEL_NAME=answerhint_2b_step62_clean_rerun DATASETS=$DS GPU_IDS=3 \
  bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_answerhint2b_clean_rerun_$(date +%Y%m%d_%H%M%S).log 2>&1
  echo "[$(date)] lane3 done"
}

lane1 & L1=$!
lane2 & L2=$!
lane3 & L3=$!
wait $L1 $L2 $L3
echo "[$(date)] Qwen3.5欠账评测+answerhint重跑 全部完成"
