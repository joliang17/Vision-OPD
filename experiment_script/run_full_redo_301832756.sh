#!/bin/bash
# 2026-07-21 终极修复版: 今天所有 eval 全灭的根因是 eval_via_vllm_server.sh 内部把 VLMEvalKit
# 的 run.py + judge 硬编码用 /usr/bin/python 跑 (不受 conda activate 影响, 只有 vllm serve 那步
# 走 PATH=conda)。/usr/bin/python 的系统栈本身还缺 termcolor 等一串 + numpy/pandas 版本互相 ABI
# 不兼容。修复: conda activate(给vllm serve) + 显式 export PYTHONPATH=.syspkg_shim(给硬编码的
# /usr/bin/python 步骤用, shim 里已补full termcolor系依赖 + 自洽的numpy2.4.6/pandas3.0.3)。
# merge 同理必须显式调用 /usr/bin/python -m verl.model_merger, 不能指望 PATH。
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=$OPSD/Vision-OPD
cd "$V"
LOG=logs/full_redo_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
EVAL_VER=server_qwen3vl2b_temp0_4096_generic
DATASETS9=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench
DATASETS7=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K
SHIM=$V/.syspkg_shim

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
VLM=$OPSD/VLMEvalKit
export PYTHONPATH=$SHIM:$VLM   # shim补硬编码/usr/bin/python缺的包; VLM根目录补 vlmeval 包本身
# (run_normal_eval.py 是以 tools/run_normal_eval.py 相对路径跑的, python只会把tools/加进
#  sys.path[0], 不会加cwd, 所以光给shim不够, 必须显式把VLMEvalKit根目录也塞进PYTHONPATH,
#  否则judge子进程 ModuleNotFoundError: No module named 'vlmeval' —— 2026-07-21 冒烟测试v2 发现)

run_lane() { # $1=gpu $2=port $3=model_path $4=model_name $5=datasets
  local gpu=$1 port=$2 mp=$3 mn=$4 ds=$5
  (
    cd "$OPSD/VLMEvalKit"
    env BACKEND=vllm_server PORT=$port EVAL_SETTING_VERSION=$EVAL_VER PYTHONPATH="$SHIM:$VLM" \
      JUDGE_API_NPROC=2 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
      MODEL_PATH=$mp MODEL_NAME=$mn DATASETS=$ds GPU_IDS=$gpu \
      bash shell_scripts/eval_model_temp0_4096.sh \
      > $V/logs/eval_local_${mn}.log 2>&1
    echo "[$(date)] lane $mn (gpu$gpu) exited rc=$?" >> "$LOG"
  ) &
}
CKPT=$V/checkpoints

# ---- 0. 补 merge FC1/FC4 中间10档 (显式系统python, 不吃 PATH) ----
log "=== stage 0: merge FC1/FC4 中间10档 (显式 /usr/bin/python) ==="
FC1=$CKPT/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
FC4=$CKPT/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
for d in "$FC1" "$FC4"; do
  for st in 10 20 40 50 70 80 100 110 130 140; do
    dd="$d/global_step_${st}"
    if [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ]; then
      /usr/bin/python -m verl.model_merger merge --backend fsdp \
        --local_dir "$dd/actor" --target_dir "$dd" >> "$LOG" 2>&1 \
        && log "merged $(basename $d)/global_step_${st}" \
        || log "MERGE STILL FAILED $(basename $d)/global_step_${st}"
    fi
  done
done
log "stage 0 merge done"

# ---- 1. 重跑 N4/S2c/QL1/alpha0/QS1 (之前全部因硬编码python坑挂零分) ----
log "=== stage 1: 重跑 5 个单点 eval ==="
run_lane 0 28950 "$CKPT/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-9B-virl39k-UNFILTERED1img-90step-trial301832756/global_step_90" n4_uniformweight_qwen35_9b_unfiltered_step90 $DATASETS9
run_lane 1 28951 "$CKPT/Vision-OPD-baseline-seed777-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832756/global_step_90" s2c_answerhint_seed777_unfiltered_step90 $DATASETS9
run_lane 2 28952 "$CKPT/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-len4096-trial301829143/global_step_90" ql1_uniform_qwen35_4b_unfiltered_len4096_step90 $DATASETS9
run_lane 3 28953 "$CKPT/Vision-OPD-contrast-alpha0-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790/global_step_90" alpha0_matched_baseline_step90 $DATASETS9
run_lane 4 28954 "$CKPT/Vision-OPD-contrast-uniform-seed1234-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-len4096-trial301832756/global_step_90" qs1_seed1234_qwen35_len4096_step90 $DATASETS9
wait
log "stage 1 done"

# ---- 2. FCE 20个之前失败的点 (merge成功的才跑) ----
log "=== stage 2: FCE 缺口20点 (8宽波次) ==="
declare -a QUEUE
for st in 10 20 40 50 70 80 100 110 130 140; do
  [ -f "$FC1/global_step_${st}/config.json" ] && QUEUE+=("fc1:$st")
  [ -f "$FC4/global_step_${st}/config.json" ] && QUEUE+=("fc4:$st")
done
n=${#QUEUE[@]}
log "FCE缺口可跑点数: $n/20"
idx=0
while [ $idx -lt $n ]; do
  batch=()
  for g in 0 1 2 3 4 5 6 7; do
    [ $idx -ge $n ] && break
    batch+=("${QUEUE[$idx]}:$g")
    idx=$((idx+1))
  done
  for item in "${batch[@]}"; do
    IFS=: read -r fam st gpu <<< "$item"
    if [ "$fam" = "fc1" ]; then d="$FC1"; mn="fc1_uniform_unfiltered_step${st}"; else d="$FC4"; mn="fc4_opsd_unfiltered_step${st}"; fi
    port=$((29100 + gpu))
    run_lane "$gpu" "$port" "$d/global_step_${st}" "$mn" "$DATASETS7"
  done
  wait
  log "FCE wave done (${#batch[@]} lanes)"
done
log "stage 2 FCE done"

log "=== full redo driver finished ==="
