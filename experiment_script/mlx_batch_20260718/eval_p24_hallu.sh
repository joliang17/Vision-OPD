set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8307 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-baseline-Qwen3-VL-4B-virl39k-UNFILTERED1img-90step-len4096-trial301783374/global_step_90" \
MODEL_NAME=answerhint_unfiltered_4b_step90 \
DATASETS=HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
