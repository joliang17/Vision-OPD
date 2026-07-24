#!/usr/bin/env bash
# 2026-07-18 用户指令: HRBench4K/8K + VStarBench 的 LLM judge 全部重判一遍
# （发现 judge 对"末行最终答案 vs 中段选项讨论"存在抽取错误, base 和训练模型都有）。
# 复用 rescore_jsd_step90.sh 配方: 低并发+高retry, 不重推理只重判分。
# 旧判分文件先备份到 prejudge_backup_20260718/ 便于 diff, 不直接删。
set -uo pipefail
cd "$(dirname "$0")/../../VLMEvalKit"
LOG="../Vision-OPD/logs/rescore_hrbench_vstar_20260718.log"
export KEY_CONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
unset no_proxy NO_PROXY
log() { echo "[$(date)] $*" | tee -a "$LOG"; }

# "model:datasets(逗号分隔)" 清单
JOBS=(
  "vanilla_qwen35_4b:HRBench4K,VStarBench"
  "vanilla_qwen35_4b_hr8k_rerun:HRBench8K"
  "contrast_std_qwen35_4b_virl39k_step90:HRBench4K,HRBench8K,VStarBench"
  "answerhint_qwen35_4b_virl39k_step90:HRBench4K,HRBench8K,VStarBench"
  "cons_qwen35_virl39k_step90_seedA:HRBench4K,HRBench8K,VStarBench"
  "cons_qwen35_virl39k_step90_seedB:HRBench4K,HRBench8K,VStarBench"
  "Qwen3-VL-4B-base-clean:HRBench4K,HRBench8K,VStarBench"
  "contrast_std_4b_virl39k_step90:HRBench4K,HRBench8K,VStarBench"
  "answerhint_4b_virl39k_step90:HRBench4K,HRBench8K,VStarBench"
)

log "=== HRBench/VStar judge 重判 started (${#JOBS[@]} models) ==="
for job in "${JOBS[@]}"; do
  M=${job%%:*}; DSS=${job##*:}
  D="outputs_vllm_curated/${M}_qwen3vl2b_temp0_4096_generic_eval/normal_scoring"
  BK="${D}/prejudge_backup_20260718"
  mkdir -p "$BK"
  for ds in ${DSS//,/ }; do
    pred="${D}/${M}_${ds}_normal.xlsx"
    [ -f "$pred" ] || { log "SKIP ${M}/${ds}: no prediction file"; continue; }
    # 备份旧判分产物再清掉（缓存在场会被静默复用）
    for f in "${D}/${M}_${ds}_normal_"*.pkl "${D}/${M}_${ds}_normal_auxmatch.xlsx" \
             "${D}/${M}_${ds}_normal_score.csv" "${D}/${M}_${ds}_normal_acc.csv" \
             "${D}/${M}_${ds}_normal_gpt-5.4-mini"*.xlsx "${D}/${M}_${ds}_normal_gpt-5.4-mini"*.log; do
      [ -f "$f" ] && mv "$f" "$BK/" 2>/dev/null
    done
    log "re-scoring ${M} / ${ds} (nproc=2, retry=12)"
    /usr/bin/python -u tools/run_normal_eval.py \
      --dataset "${ds}" \
      --prediction-file "${pred}" \
      --judge gpt-5.4-mini-2026-03-17 \
      --provider tiktok_azure \
      --nproc 2 \
      --retry 12 \
      --timeout 600 \
      --temperature 0.0 \
      --key-conf "${KEY_CONF}" \
      >> "$LOG" 2>&1 || log "WARNING: ${M}/${ds} rescore failed"
    new=$(head -2 "${D}/${M}_${ds}_normal_acc.csv" 2>/dev/null | tail -1)
    old=$(head -2 "${BK}/${M}_${ds}_normal_acc.csv" 2>/dev/null | tail -1)
    log "RESULT ${M}/${ds}: old=[${old}] new=[${new}]"
  done
done
log "=== 重判 done ==="
