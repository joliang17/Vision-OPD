#!/bin/bash
# M1 (arXiv plan): visual-reliance motivation probe. 1 GPU, ~1h for 500x2x2 generations.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

V=${V:-/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD}
LMU=${LMU:-/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/LMUData}
CACHE=${CACHE:-/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache}
DEFAULT_BASE2B=$CACHE/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
BASE2B=${BASE2B:-$DEFAULT_BASE2B}
NUM_SAMPLES=${NUM_SAMPLES:-500}
OUTPUT_DIR=${OUTPUT_DIR:-analysis_outputs/m1_probe}
DEVICE=${DEVICE:-cuda}

cd "$V"
mkdir -p logs "$OUTPUT_DIR"
test -f "$LMU/POPE.tsv"
test -f "$LMU/VStarBench.tsv"

python3 -u scripts/m1_visual_reliance_probe.py --model-path "$BASE2B" \
  --tsv "$LMU/POPE.tsv" --dataset-type yesno --num-samples "$NUM_SAMPLES" --device "$DEVICE" \
  --output "$OUTPUT_DIR/pope_2b_base.json" 2>&1 | tee logs/m1_pope_2b.log
python3 -u scripts/m1_visual_reliance_probe.py --model-path "$BASE2B" \
  --tsv "$LMU/VStarBench.tsv" --dataset-type mcq --num-samples "$NUM_SAMPLES" --device "$DEVICE" \
  --output "$OUTPUT_DIR/vstar_2b_base.json" 2>&1 | tee logs/m1_vstar_2b.log
