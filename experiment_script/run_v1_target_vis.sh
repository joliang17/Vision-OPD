#!/bin/bash
# V1 (arXiv plan): target-distribution decoding visualization. 1 GPU.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
BASE2B=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
python3 scripts/visualize_target_decoding.py \
  --teacher-path "$BASE2B" \
  --rollout-dir rollouts/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step \
  --parquet data/virl39k_train_noimg_filtered_1img.parquet \
  --max-samples 12 --alpha 1.0 --beta 0.1 --ctrl-mode black --topk 5 \
  --output analysis_outputs/target_decoding_vis/report_alpha1.0_beta0.1_black.md \
  2>&1 | tee logs/v1_target_vis_$(date +%Y%m%d_%H%M%S).log
