set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
# judge 并发压低+重试拉高：2026-07-16 Azure judge 有过 429 限流事故(JSD那次MCQ分数全废)
PATH=$V/scripts/qwen35_shim:$PATH BACKEND=vllm_server PORT=8266 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 \
MODEL_PATH=${V}/checkpoints/Vision-OPD-grpo-baseline-default-Qwen3.5-4B-trial301761390/global_step_195 \
MODEL_NAME=grpo_baseline_default_qwen35_4b_step195 \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
