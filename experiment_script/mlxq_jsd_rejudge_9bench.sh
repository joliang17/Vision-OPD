set -euo pipefail
# JSD step90 重判分:原判分被429降级为exact-match全废(BLINK 42.14实为~57.5)
# 预测文件保留,污染的判分产物已删,reuse只重跑judge
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
JUDGE_API_NPROC=3 JUDGE_RETRY=12 PORT=8300 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-jsd-Qwen3-VL-2B-virl39k-90step-trial301761390/global_step_90" \
MODEL_NAME=jsd_2b_virl39k_90step_step90 \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh
