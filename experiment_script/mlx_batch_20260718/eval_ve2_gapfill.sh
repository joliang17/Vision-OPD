set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
PATH=$V/scripts/qwen35_shim:$PATH BACKEND=vllm_server PORT=8303 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH=${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90 \
MODEL_NAME=unfiltered_qwen35_virl39k_step90 \
DATASETS=HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
