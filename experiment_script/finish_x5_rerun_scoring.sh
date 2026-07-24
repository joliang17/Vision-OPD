#!/usr/bin/env bash
# X5 rerun 收尾: 推理缓存 pkl -> xlsx -> 判分（前两次 score 空跳的根因: run.py reuse 模式不重写 xlsx）
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "$OPSD/VLMEvalKit"
LOG="$OPSD/Vision-OPD/logs/finish_x5_scoring.log"
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
M=contrast_std_4b_virl39k_step90_rerun0718
W=outputs_vllm_curated/${M}_qwen3vl2b_temp0_4096_generic_eval
TD=$(ls -d $W/$M/T*/ | sort | tail -1)
export KEY_CONF=$OPSD/config/key.conf
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
unset no_proxy NO_PROXY
SHIM=$OPSD/Vision-OPD/.syspkg_shim

log "=== pkl -> xlsx (T-dir: $TD) ==="
PYTHONNOUSERSITE=1 PYTHONPATH=$SHIM /usr/bin/python - "$TD" "$M" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, types, pickle, glob, os
import pandas as pd
from pandas import Index
mod=types.ModuleType('pandas.core.indexes.numeric')
class _C:
    def __new__(cls, data=None, dtype=None, name=None, **k): return Index([] if data is None else data, dtype=dtype, name=name)
for n in ['Int64Index','Float64Index','UInt64Index','NumericIndex']: setattr(mod,n,_C)
sys.modules['pandas.core.indexes.numeric']=mod
td, m = sys.argv[1], sys.argv[2]
for p in sorted(glob.glob(f'{td}/{m}_*.pkl')):
    ds=os.path.basename(p)[len(m)+1:-4]
    if ds=='status': continue
    d=pickle.load(open(p,'rb'))
    out=f'{td}/{m}_{ds}.xlsx'
    d.to_excel(out, index=False)
    print(f'{ds}: {len(d)} rows -> {out}')
PYEOF

mkdir -p $W/normal_scoring
for ds in BLINK MMStar MMBench_DEV_EN VStarBench MathVista_MINI HRBench4K HRBench8K POPE HallusionBench; do
  src=$TD/${M}_${ds}.xlsx
  [ -f "$src" ] || { log "SKIP $ds no xlsx"; continue; }
  pred=$W/normal_scoring/${M}_${ds}_normal.xlsx
  cp -f "$src" "$pred"
  log "scoring $ds"
  /usr/bin/python -u tools/run_normal_eval.py \
    --dataset "$ds" --prediction-file "$pred" \
    --judge gpt-5.4-mini-2026-03-17 --provider tiktok_azure \
    --nproc 2 --retry 12 --timeout 600 --temperature 0.0 \
    --key-conf "$KEY_CONF" >> "$LOG" 2>&1 || log "WARNING: $ds scoring failed"
done
log "=== X5 scoring finish done ==="
