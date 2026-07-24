#!/bin/bash
# 重启#2（vlmeval 依赖装齐后）：只重启四条评测泳道；GPU0-3 的训练泳道还活着不动
# native 8k 重推已成功(73.75)不再跑
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

ninebench() {  # <ckpt> <tag> <gpu> [datasets]
  local DS="${4:-BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench}"
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server MODEL_PATH="$1" MODEL_NAME="$2" DATASETS="$DS" GPU_IDS=$3 \
  bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/r3_eval_$2_$(date +%Y%m%d_%H%M%S).log" 2>&1
  cd "$V"
}

lane_gpu4() {
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server \
  MODEL_PATH=$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct/global_step_62 \
  MODEL_NAME=native_setting_std_step62_8192_pp0 \
  DATASETS=VStarBench,HRBench4K,HRBench8K MAX_NEW_TOKENS=8192 PRESENCE_PENALTY=0 GPU_IDS=4 \
  bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/r3_native_setting_eval_$(date +%Y%m%d_%H%M%S).log" 2>&1
  cd "$V"
  for s in 30 60 90; do
    ninebench "$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered-90step/global_step_$s" "std_sr1_90step_step$s" 4
  done
  echo "[$(date)] r3 lane_gpu4 完成"
}

lane_gpu5() {
  ninebench $V/checkpoints/Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62 uniform_weight_hrbench8k_rerun 5 HRBench8K
  ninebench $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-Instruct/global_step_62 std4b_hrbench_rerun 5 HRBench4K,HRBench8K
  ninebench $V/checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-4B-Instruct/global_step_62 cons4b_hrbench_rerun 5 HRBench4K,HRBench8K
  ninebench $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered/global_step_140 std_sr1_step140_hrbench8k_rerun 5 HRBench8K
  bash scripts/run_zoombench_canonical.sh \
    checkpoints/Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62 \
    contrast_standard_uniform_weight_step62 5 8045 > logs/r3_zoombench_uniform.log 2>&1
  echo "[$(date)] r3 lane_gpu5 完成"
}

lane_gpu6() {
  ninebench $V/checkpoints/Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered-3ep/global_step_1392 grpo_virl39k_3ep_step1392 6
  ninebench $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30 std_virl39k_90step_step30 6
  ninebench $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_60 std_virl39k_90step_step60 6
  bash scripts/run_zoombench_canonical.sh \
    checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30 \
    contrast_standard_virl39k_90step_step30_rerun 6 8044 > logs/r3_zoombench_std30_rerun.log 2>&1
  echo "[$(date)] r3 lane_gpu6 完成"
}

lane_gpu7() {
  for s in 30 60 90; do
    ninebench $V/checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_$s cons_virl39k_90step_step$s 7
  done
  echo "[$(date)] r3 lane_gpu7 完成"
}

lane_gpu4 & P2=$!
lane_gpu5 & P3=$!
lane_gpu6 & P4=$!
lane_gpu7 & P5=$!
wait $P2 $P3 $P4 $P5
echo "[$(date)] r3 全部评测泳道完成"
