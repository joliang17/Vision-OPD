set -euo pipefail
# 补跑任务：HRBench8K 之前因 HRBench8K_local.tsv 图片路径写死 /home/tiger（容器内不存在）被
# 静默跳过，图片已拷到 NAS 且 tsv 路径已修复（2026-07-15）。此脚本按四组采样设置补跑
# HRBench8K（MODEL_NAME 沿用原名，结果落回各自已有输出目录），最后跑 ZoomBench canonical。

OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CKPT_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30

cd "${OPSD_ROOT}/Vision-OPD"
if [ ! -f "${CKPT_DIR}/config.json" ]; then
  bash scripts/merge_checkpoint.sh "${CKPT_DIR}"
fi

cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt

MODEL_PATH_ABS="${OPSD_ROOT}/Vision-OPD/${CKPT_DIR}"

# 1) 默认设置（temp=0, presence=1.5, 4096）
MODEL_PATH="${MODEL_PATH_ABS}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30 \
DATASETS=HRBench8K GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh

# 2) 采样版（temp=0.7, top_p=0.8, top_k=20, presence=1.5, 16k）
MODEL_PATH="${MODEL_PATH_ABS}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_t07_16k \
DATASETS=HRBench8K GPU_IDS=0 \
TEMPERATURE=0.7 TOP_P=0.8 TOP_K=20 PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=16384 \
bash shell_scripts/eval_via_vllm_server.sh

# 3) greedy 16k（temp=0, presence=1.5, 16k）
MODEL_PATH="${MODEL_PATH_ABS}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_greedy_16k \
DATASETS=HRBench8K GPU_IDS=0 \
TEMPERATURE=0.0 PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=16384 \
bash shell_scripts/eval_via_vllm_server.sh

# 4) greedy 无 presence（temp=0, presence=0, 32k）
MODEL_PATH="${MODEL_PATH_ABS}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_greedy_p0_32k \
DATASETS=HRBench8K GPU_IDS=0 \
TEMPERATURE=0.0 PRESENCE_PENALTY=0.0 REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=32768 \
bash shell_scripts/eval_via_vllm_server.sh

# 5) ZoomBench canonical（infer 端 temp=0 写死；与完整版任务同 tag，结果覆盖/复核 40.95%）
cd "${OPSD_ROOT}/Vision-OPD"
bash scripts/run_zoombench_canonical.sh \
  "${CKPT_DIR}" \
  contrast_standard_virl39k_90step_step30 0 8010
