#!/bin/bash
# contrast-标准/保守×Qwen3.5 的 VLMEvalKit 9-bench 主口径评测(conda shim)
# 标准版已ready; 保守版等训完+merge。等 geo3k 训练(GPU1,2,3,6)结束后用 GPU1/GPU2
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
SHIM=$V/scripts/qwen35_shim
DS="BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench"

echo "[$(date)] 等 geo3k 训练结束..."
while pgrep -f "run_geo3k_contrast_4b.sh" >/dev/null 2>&1; do sleep 120; done

lane_std() {
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  PATH=$SHIM:$PATH BACKEND=vllm_server PORT=28831 \
  MODEL_PATH=$V/checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-trial301683547/global_step_62 \
  MODEL_NAME=contrast_std_qwen35_4b_step62 DATASETS=$DS GPU_IDS=1 \
  bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_contrast_std_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
  # ZoomBench (canonical, conda shim vllm serve由canonical脚本自己起——canonical用vllm serve? 检查过它直接vllm serve, shim生效)
  cd $V
  PATH=$SHIM:$PATH bash scripts/run_zoombench_canonical.sh \
    checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-trial301683547/global_step_62 \
    contrast_std_qwen35_4b_step62 1 28833 > logs/zoombench_contrast_std_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
  echo "[$(date)] contrast-std Qwen3.5 评测完成"
}
lane_cons() {
  CONS=$V/checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-trial301683547
  while true; do
    FS=$(cat $CONS/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
    [ "$FS" -ge 62 ] && [ -f "$CONS/global_step_$FS/config.json" ] && break
    sleep 300
  done
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  PATH=$SHIM:$PATH BACKEND=vllm_server PORT=28832 \
  MODEL_PATH=$CONS/global_step_$FS \
  MODEL_NAME=contrast_cons_qwen35_4b_step$FS DATASETS=$DS GPU_IDS=2 \
  bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_contrast_cons_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
  cd $V
  PATH=$SHIM:$PATH bash scripts/run_zoombench_canonical.sh \
    $CONS/global_step_$FS contrast_cons_qwen35_4b_step$FS 2 28834 > logs/zoombench_contrast_cons_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
  echo "[$(date)] contrast-cons Qwen3.5 评测完成"
}
lane_std & P1=$!
lane_cons & P2=$!
wait $P1 $P2
echo "[$(date)] contrast×Qwen3.5 VLMEvalKit评测全部完成"
