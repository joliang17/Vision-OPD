set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
PATH=$V/scripts/qwen35_shim:$PATH BACKEND=vllm_server PORT=8278 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301761390/global_step_90 \
MODEL_NAME=cons_qwen35_virl39k_step90_seedA \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
