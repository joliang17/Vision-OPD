set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8282 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-Instruct-virl39k-filtered-90step-trial301829143/global_step_90" \
MODEL_NAME=answerhint_2b_virl39k_90step_step90 \
DATASETS=MathVerse_MINI,WeMath GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
