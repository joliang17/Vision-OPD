#!/usr/bin/env bash
# 2026-07-18 用户指令: X5 (4B contrast-std virl39k step90) 分数低得不对劲, 本机重跑完整 eval+judge。
# 口径对齐: PYTHONNOUSERSITE=1 → 系统栈 vllm 0.11.0 (与原 mlx 容器同版本, 规避 0.11/0.18 栈差 ~1.9pp)。
# MODEL_NAME 加 _rerun0718 后缀, 与原结果分目录, 方便 diff。
# judge 低并发 (nproc=2): 与后台 HRBench/VStar 重判批次共享 Azure 端点, 防止互相打出 429。
set -uo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
LOG="${OPSD_ROOT}/Vision-OPD/logs/rerun_x5_eval_local.log"
log() { echo "[$(date)] $*" | tee -a "$LOG"; }

export PYTHONNOUSERSITE=1
export PYTHONPATH="${OPSD_ROOT}/Vision-OPD/.syspkg_shim${PYTHONPATH:+:$PYTHONPATH}"  # 系统栈缺 termcolor/num2words/ijson 的补丁目录
# 本地 vllm serve 走 localhost, judge 走代理
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
export KEY_CONF="${OPSD_ROOT}/config/key.conf"

log "=== X5 local rerun started (system stack vllm 0.11, GPU0) ==="
BACKEND=vllm_server PORT=8291 \
JUDGE_API_NPROC=2 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-filtered-90step-trial301761390/global_step_90" \
MODEL_NAME=contrast_std_4b_virl39k_step90_rerun0718 \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh >> "$LOG" 2>&1
rc=$?
log "=== X5 local rerun finished rc=$rc ==="
