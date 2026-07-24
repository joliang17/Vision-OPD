#!/bin/bash
# 14 个 FCE 点重推 Hallu（连推理产物都没有），等 uniform β=0 训完腾卡后批量跑
# 每点 1 卡 serve + 判分；8 卡并行，两批跑完 14 点
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd; V=$OPSD/Vision-OPD; cd $V
LOG=logs/fce_hallu_reinfer_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
all_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }
BETA0=checkpoints/Vision-OPD-contrast-uniform-beta0-Qwen3-VL-2B-virl39k-UNFILTERED1img-ext200-trial301967423
log "armed: 等 uniform β=0 训完(step200 或 driver 结束)+ 8卡空"
# 等 β=0 到 200 或它的 driver 不在了（崩了也算腾卡）
while pgrep -f run_uniform_beta0_ext200 >/dev/null && [ "$(cat $BETA0/latest_checkpointed_iteration.txt 2>/dev/null||echo 0)" -lt 200 ]; do sleep 180; done
until all_free; do sleep 120; done; sleep 30

FC1=$V/checkpoints/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
FC4=$V/checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
# (ckpt目录 名前缀 step)
PTS=(
 "$FC1 fc1_uniform_unfiltered 100" "$FC1 fc1_uniform_unfiltered 110" "$FC1 fc1_uniform_unfiltered 130" "$FC1 fc1_uniform_unfiltered 140"
 "$FC4 fc4_opsd_unfiltered 10" "$FC4 fc4_opsd_unfiltered 20" "$FC4 fc4_opsd_unfiltered 40" "$FC4 fc4_opsd_unfiltered 50"
 "$FC4 fc4_opsd_unfiltered 70" "$FC4 fc4_opsd_unfiltered 80" "$FC4 fc4_opsd_unfiltered 100" "$FC4 fc4_opsd_unfiltered 110"
 "$FC4 fc4_opsd_unfiltered 130" "$FC4 fc4_opsd_unfiltered 140"
)
cd $OPSD/VLMEvalKit
export PYTHONPATH=$V/.syspkg_shim_nonumpy:$OPSD/VLMEvalKit
i=0
for spec in "${PTS[@]}"; do
  set -- $spec; ckdir=$1; name=$2; st=$3
  gpu=$((i%8)); port=$((30500+i))
  (
    CK=$ckdir/global_step_$st
    BACKEND=vllm_server PORT=$port MODEL_PATH=$CK MODEL_NAME=${name}_step${st} \
    DATASETS=HallusionBench GPU_IDS=$gpu JUDGE_API_NPROC=4 REQUEST_TIMEOUT=900 RETRY=12 \
    bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/reinfer_${name}_step${st}_hallu.log 2>&1
    echo "[$(date +%H:%M)] ${name}_step${st} done" >> $V/logs/fce_hallu_reinfer_driver.log
  ) &
  i=$((i+1))
  [ $((i%8)) -eq 0 ] && wait   # 8 卡一批
done
wait
log "=== 14点 Hallu 重推 done ==="
