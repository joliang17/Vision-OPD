set -euo pipefail
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Zooming-without-Zooming/mm-eval
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1" MKL_SERVICE_FORCE_INTEL=1
SNAP=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
python3 infer_without_tool.py --benchmark mme-realworld-cn --model_path "$SNAP" --model qwen3vl4b_base_official_repro \
  --gpus 1 --temperature 0.7 --seed 42 --gpu_memory_utilization 0.85 --max_model_len 24576
python3 judge_azure.py --benchmark mme-realworld-cn --model qwen3vl4b_base_official_repro_seed42 \
  --key_conf /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
