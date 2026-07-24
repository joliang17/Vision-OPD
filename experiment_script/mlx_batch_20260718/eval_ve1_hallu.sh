set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8306 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-UNFILTERED1img-90step-trial301783374/global_step_90" \
MODEL_NAME=unfiltered_4b_virl39k_step90 \
DATASETS=HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
