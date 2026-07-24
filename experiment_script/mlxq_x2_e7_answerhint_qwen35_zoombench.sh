set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
CKPT=$V/checkpoints/Vision-OPD-baseline-Qwen3.5-4B-virl39k-filtered-trial301761390/global_step_90
# 等9-bench任务(mlxq_x2..._9bench)先完成merge;若并发启动则最多等40分钟
for i in $(seq 1 40); do [ -f "${CKPT}/config.json" ] && break; sleep 60; done
[ -f "${CKPT}/config.json" ] || { echo "merge not ready after 40min"; exit 1; }
cd "$V"
PATH=$V/scripts/qwen35_shim:$PATH bash scripts/run_zoombench_canonical.sh \
  "$CKPT" answerhint_qwen35_4b_virl39k_step90 0 8281
