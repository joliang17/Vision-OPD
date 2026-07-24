set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8310 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B \
MODEL_NAME=vanilla_qwen35_4b_nothink \
DATASETS=HRBench4K,HRBench8K,POPE,VStarBench,BLINK,MMStar GPU_IDS=0 \
ENABLE_THINKING=false \
bash shell_scripts/eval_model_temp0_4096_nothink.sh
