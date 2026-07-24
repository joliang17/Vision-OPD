set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8283 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90" \
MODEL_NAME=contrast_std_unfiltered1img_2b_step90_server \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
