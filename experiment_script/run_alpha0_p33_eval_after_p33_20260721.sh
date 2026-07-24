#!/bin/bash
# α=0 + P33 的 eval，挂在 P33 训练之后（2026-07-21，301832790）：
#   P33 抢先占了 8 卡训练，α=0 已训完+merge 但 eval 被挤后。等 P33 训完（step90 merged）+ GPU 空，
#   4 路并行：α0 9-bench(GPU0) / α0 Zoom(GPU1) / P33 9-bench(GPU2) / P33 Zoom(GPU3)。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "$V"
LOG=logs/alpha0_p33_eval_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

A0=checkpoints/Vision-OPD-contrast-alpha0-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790/global_step_90
P33=checkpoints/Vision-OPD-contrast-noanchor-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790
log "armed: waiting for P33 training done (step90 merged)"
while [ ! -f "$P33/global_step_90/config.json" ]; do sleep 300; done
log "P33 step90 merged; waiting a bit for GPUs to settle"; sleep 60

run9(){ # gpu port ckpt name
  cd "$OPSD/VLMEvalKit"
  BACKEND=vllm_server PORT=$2 MODEL_PATH="$3" MODEL_NAME="$4" \
  DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
  GPU_IDS=$1 JUDGE_API_NPROC=3 REQUEST_TIMEOUT=900 RETRY=12 \
  bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/${4}_9bench.log" 2>&1
  cd "$V"; log "$4 9-bench done"
}

( run9 0 29000 "$V/$A0" alpha0_matched_2b_step90 ) &
( run9 2 29002 "$V/$P33/global_step_90" p33_noanchor_2b_step90 ) &
# Zoom 两路（复用 g1 zoom driver 模式：各自 serve+infer+judge）
( GPU=1 PORT=29001 CKPT="$V/$A0" NAME=alpha0_matched_2b_step90 bash scripts/_zoom_eval_generic.sh ) &
( GPU=3 PORT=29003 CKPT="$V/$P33/global_step_90" NAME=p33_noanchor_2b_step90 bash scripts/_zoom_eval_generic.sh ) &
wait
log "=== α0 + P33 eval driver finished ==="
