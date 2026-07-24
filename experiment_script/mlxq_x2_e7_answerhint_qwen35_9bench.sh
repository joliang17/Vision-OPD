set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
CKPT=$V/checkpoints/Vision-OPD-baseline-Qwen3.5-4B-virl39k-filtered-trial301761390/global_step_90
# merge guard: step90 还没merge(只有step145 merge过),qwen3_5架构需要conda里的transformers 5.5
if [ ! -f "${CKPT}/config.json" ]; then
  cd "$V"
  conda run -n qwen35 --cwd "$V" python3 -m verl.model_merger merge \
    --backend fsdp --local_dir "${CKPT}/actor" --target_dir "${CKPT}"
fi
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
PATH=$V/scripts/qwen35_shim:$PATH BACKEND=vllm_server PORT=8280 \
JUDGE_API_NPROC=3 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH=${CKPT} \
MODEL_NAME=answerhint_qwen35_4b_virl39k_step90 \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
