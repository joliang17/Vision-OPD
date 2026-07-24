set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
JUDGE_API_NPROC=3 JUDGE_RETRY=12 PORT=8269 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered/global_step_464" \
MODEL_NAME=grpo-2B-virl39kfiltered-step464_server \
DATASETS=MathVerse_MINI,WeMath GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh
