#!/bin/bash
# VDH 视觉依赖高亮 批量(Qwen3.5-2B base/opsd/ours,12样本),GPU0,conda qwen35
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=0
BASE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-2B
OPSD=$V/checkpoints/Vision-OPD-baseline-Qwen3.5-2B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90
OURS=$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-2B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90
python3 scripts/visual_dependency_highlight.py \
  --manifest docs/vdh_manifest.jsonl \
  --base "$BASE" --opsd "$OPSD" --ours "$OURS" \
  --control black --out-dir docs/vdh_out
echo "=== VDH DONE rc=$? ==="
