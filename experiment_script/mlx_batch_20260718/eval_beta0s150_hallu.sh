set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8309 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-90step-trial301783374/global_step_150" \
MODEL_NAME=beta0_2b_virl39k_step150 \
DATASETS=HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
