set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Zooming-without-Zooming/mm-eval
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1" MKL_SERVICE_FORCE_INTEL=1
for BENCH in vstar hrbench-4k hrbench-8k; do
  conda run -n qwen35 python3 infer_without_tool.py --benchmark $BENCH --model_path /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-visionopd-Qwen3.5-4B-trial301761390/global_step_62 --model visionopd_qwen35_official_repro \
    --gpus 1 --temperature 0.7 --seed 42 --gpu_memory_utilization 0.85 --max_model_len 24576
  /usr/bin/python3 judge_azure.py --benchmark $BENCH --model visionopd_qwen35_official_repro_seed42 \
    --key_conf /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
done
