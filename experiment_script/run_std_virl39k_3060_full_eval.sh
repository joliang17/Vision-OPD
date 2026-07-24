#!/bin/bash
# 标准×virl39k-90step step30/60：9-benchmark(server模式) + ZoomBench canonical
# merge 已提前完成。GPU5 跑 step30，GPU6 跑 step60（并行），各自评完9-bench后接自己的ZoomBench
set -x

wait_for_grpo() {
  while pgrep -f "trainer.experiment_name=Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered-3ep" >/dev/null 2>&1; do
    sleep 30
  done
}

run_one() {
  local STEP=$1 GPU=$2 PORT=$3
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  MODEL_PATH=../Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_${STEP} \
  MODEL_NAME=contrast_standard_virl39k_90step_step${STEP} \
  DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
  GPU_IDS=${GPU} \
  bash shell_scripts/eval_via_vllm_server.sh \
    > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_std_virl39k_90step_step${STEP}_301783374_$(date +%Y%m%d_%H%M%S).log 2>&1

  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
  bash scripts/run_zoombench_canonical.sh \
    checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_${STEP} \
    contrast_standard_virl39k_90step_step${STEP} ${GPU} ${PORT} \
    > logs/zoombench_std_virl39k_90step_step${STEP}_301783374_$(date +%Y%m%d_%H%M%S).log 2>&1
}

echo "[$(date)] 等待 grpo-virl39k-3ep 训练进程结束..."
wait_for_grpo
echo "[$(date)] grpo-3ep 已结束，GPU5跑step30 / GPU6跑step60 并行启动"

run_one 30 5 8010 &
P30=$!
run_one 60 6 8011 &
P60=$!
wait $P30
wait $P60
echo "[$(date)] 标准×virl39k-90step step30/60 全部评测(9-bench+ZoomBench)完成"
