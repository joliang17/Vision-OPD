#!/bin/bash
# trial 301761390: P31 — β=0 边界续训 90→200 步（2026-07-18 用户下达）
# resume P13 的 ckpt（Vision-OPD-contrast-beta0-...-trial301783374, latest=90, 分片 world_size_4 ⇒ 必须 4 卡续）
# 与 P21（β=0.1→200, 301832790 已跑）两臂同步数对比崩溃起点。
# 判定规则（用户写死）：150-200 区间与 β=0.1 同样健康 → β 删；更早恶化 → β 留。
# 附带: 续训段 rollout dump 复读/中英混杂扫描。eval step150/180/200 各 9-bench 由 mlx 提交（queue.md 已排）。
# 等 X5 本机重跑（GPU0）结束后再启动；训练用 GPU1-4 避开。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p31_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
NAME=Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-90step-trial301783374
CKPT=checkpoints/$NAME
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet

log "=== P31 driver armed: waiting for X5 local rerun to finish ==="
while [ "$(grep -c 'local rerun finished' logs/rerun_x5_eval_local.log 2>/dev/null || echo 0)" -lt 3 ]; do sleep 300; done
log "X5 rerun done; checking GPUs 1-4 free"
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2,3,4 | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done

s0=$(cat $CKPT/latest_checkpointed_iteration.txt)
[ "$s0" -ne 90 ] && { log "FATAL: expected step90, got $s0 — 有人动过这个 run? 停止"; exit 1; }
TLOG=logs/p31_beta0_extend200_$(date +%Y%m%d_%H%M%S).log
log "resuming $NAME 90->200 on GPU1-4 (world_size_4 匹配)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=1,2,3,4 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=200 \
  > "$TLOG" 2>&1 &
sleep 60
while true; do
  s=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  [ "${s:-0}" -ge 200 ] && break
  age=$(( $(date +%s) - $(stat -c %Y "$TLOG" 2>/dev/null || date +%s) ))
  if [ "$age" -gt 900 ] && ! ps aux | grep -v grep | grep -q verl.trainer.main_ppo; then
    log "DEAD at step ${s} — 看 $TLOG"; exit 1
  fi
  sleep 300
done
log "step200 reached; merging 150/180/200"
for st in 150 180 200; do
  dd="$CKPT/global_step_${st}"
  [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
    bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "step${st} merged"
done

log "rollout 复读/中英混杂扫描 (step>90 的 dump)"
python3 - <<'PYEOF' >> "$LOG" 2>&1
import glob, json, re, os
import pandas as pd
base='rollouts/Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-90step-trial301783374'
rows=[]
for f in sorted(glob.glob(f'{base}/*')):
    m=re.search(r'(\d+)', os.path.basename(f))
    if not m: continue
    step=int(m.group(1))
    if step<=90: continue
    try:
        if f.endswith('.jsonl'):
            texts=[json.loads(l).get('output','') or json.loads(l).get('response','') for l in open(f)][:256]
        elif f.endswith('.parquet'):
            d=pd.read_parquet(f); col=[c for c in d.columns if c in ('output','response','responses','generations')]
            texts=d[col[0]].astype(str).tolist()[:256] if col else []
        else: continue
    except Exception as e:
        print('skip',f,e); continue
    def rep_ratio(t):
        toks=str(t).split()
        if len(toks)<20: return 0.0
        tri=[' '.join(toks[i:i+3]) for i in range(len(toks)-2)]
        return 1-len(set(tri))/len(tri)
    def cjk(t):
        s=str(t); c=sum('一'<=ch<='鿿' for ch in s)
        return c/max(len(s),1)
    reps=[rep_ratio(t) for t in texts]; cj=[cjk(t) for t in texts]
    rows.append(dict(step=step,n=len(texts),rep_mean=sum(reps)/max(len(reps),1),
                rep_p95=sorted(reps)[int(0.95*len(reps))] if reps else 0,
                frac_rep_gt_half=sum(r>0.5 for r in reps)/max(len(reps),1),
                cjk_mean=sum(cj)/max(len(cj),1)))
out=pd.DataFrame(rows).sort_values('step')
os.makedirs('analysis_outputs/p31_beta0_extend_scan',exist_ok=True)
out.to_csv('analysis_outputs/p31_beta0_extend_scan/repetition_scan.csv',index=False)
print(out.to_string())
PYEOF
log "=== P31 driver done ==="
