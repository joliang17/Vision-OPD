#!/bin/bash
# 补跑 task5 缺失的评测：301638440 上的 task5 eval driver 因 cwd 错误 8/9 全部瞬间失败
# （日志只有一行 "shell_scripts/eval_model_temp0_4096.sh: No such file or directory"），
# 只有 std_virl39k step60 真正跑过。本脚本补：
#   cons_virl39k step30/60/90 + std_sr1 step30/60/90 的 9-benchmark + ZoomBench canonical
#   （检查过6个checkpoint都已merge）
# 外加 uniform-weight step62 的 ZoomBench（之前只跑过9-bench）
# 三条静态泳道 GPU5/6/7，各自等本机之前排的任务脚本退出后接手，不动态抢卡
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

CONS_VIRL=checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step
STD_SR1=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered-90step
UNIFORM=checkpoints/Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62

run_one() {  # <ckpt_dir> <tag> <gpu> <port>
  local CKPT=$1 TAG=$2 GPU=$3 PORT=$4
  if [ ! -f "${CKPT}/config.json" ]; then
    echo "[$(date)] SKIP ${TAG}: 未merge" >> logs/task5_backfill_skips.log
    return
  fi
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server \
  MODEL_PATH="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/${CKPT}" \
  MODEL_NAME="${TAG}" \
  DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
  GPU_IDS=${GPU} \
  bash shell_scripts/eval_model_temp0_4096.sh \
    > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/backfill_eval_${TAG}_$(date +%Y%m%d_%H%M%S).log 2>&1
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
  bash scripts/run_zoombench_canonical.sh "${CKPT}" "${TAG}" ${GPU} ${PORT} \
    > logs/backfill_zoombench_${TAG}_$(date +%Y%m%d_%H%M%S).log 2>&1
}

zoombench_only() {  # <ckpt_dir> <tag> <gpu> <port>
  local CKPT=$1 TAG=$2 GPU=$3 PORT=$4
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
  bash scripts/run_zoombench_canonical.sh "${CKPT}" "${TAG}" ${GPU} ${PORT} \
    > logs/backfill_zoombench_${TAG}_$(date +%Y%m%d_%H%M%S).log 2>&1
}

lane_gpu5() {
  # 等 std_virl39k step30/60 的driver退出（它占GPU5/6）
  while pgrep -f "run_std_virl39k_3060_full_eval.sh" >/dev/null 2>&1; do sleep 60; done
  run_one "${CONS_VIRL}/global_step_30" cons_virl39k_90step_step30 5 8020
  run_one "${CONS_VIRL}/global_step_60" cons_virl39k_90step_step60 5 8021
}

lane_gpu6() {
  while pgrep -f "run_std_virl39k_3060_full_eval.sh" >/dev/null 2>&1; do sleep 60; done
  run_one "${CONS_VIRL}/global_step_90" cons_virl39k_90step_step90 6 8022
  run_one "${STD_SR1}/global_step_30" std_sr1_90step_step30 6 8023
}

lane_gpu7() {
  # 等 grpo3ep eval driver 退出（它占GPU7）
  while pgrep -f "run_grpo3ep_eval_after_training.sh" >/dev/null 2>&1; do sleep 60; done
  run_one "${STD_SR1}/global_step_60" std_sr1_90step_step60 7 8024
  run_one "${STD_SR1}/global_step_90" std_sr1_90step_step90 7 8025
  zoombench_only "${UNIFORM}" contrast_standard_uniform_weight_step62 7 8026
}

lane_gpu5 &
L5=$!
lane_gpu6 &
L6=$!
lane_gpu7 &
L7=$!
wait $L5 $L6 $L7
echo "[$(date)] task5 缺失评测补跑全部完成"
