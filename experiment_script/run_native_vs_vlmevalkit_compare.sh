#!/bin/bash
# 对比 Vision-OPD 原生eval(infer.py->judge_qwenlm.py->cal_acc.py) vs VLMEvalKit 在同一个checkpoint上的分数
# 用 contrast-标准-Qwen3-VL-2B-Instruct/global_step_62（有已知VLMEvalKit分数可比：vstar 76.96/hrbench4k 77.62/hrbench8k 73.75）
# 挑 vstar/hrbench-4k/hrbench-8k 三个两边都有的benchmark
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

echo "[$(date)] 【顺序2/2】等 VLMEvalKit原生配置版eval(run_vlmevalkit_native_setting_eval.sh)先跑完..."
while pgrep -f "run_vlmevalkit_native_setting_eval.sh" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date)] 原生配置版已结束，开始起vllm serve跑原生eval"

CKPT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct/global_step_62
SERVED_NAME=native_compare_std_step62
PORT=8010

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# 固定用 GPU4（此时另一个收尾脚本的保守×sr1-90step 训练占的是GPU0-3，这里避开不撞车）
export CUDA_VISIBLE_DEVICES=4
nohup vllm serve "${CKPT}" \
  --served-model-name "${SERVED_NAME}" --host 0.0.0.0 --port ${PORT} \
  --trust-remote-code --enforce-eager --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --dtype bfloat16 --limit-mm-per-prompt '{"image":8}' \
  > Vision-OPD/logs/native_compare_vllm_serve_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SERVER_PID=$!

echo "[$(date)] 等 vllm server 起来..."
until curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  sleep 10
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[$(date)] vllm server 进程死了，中止"
    exit 1
  fi
done
echo "[$(date)] vllm server 就绪，开始跑 native eval"

cd Vision-OPD/eval
for BENCH in vstar hrbench-4k hrbench-8k; do
  case "$BENCH" in
    vstar) JSON=vstar.json ;;
    hrbench-4k) JSON=hr_bench_4k.json ;;
    hrbench-8k) JSON=hr_bench_8k.json ;;
  esac
  python3 infer.py --benchmark "${BENCH}" --benchmark_json "$PWD/${JSON}" \
    --out_dir model_answer --model_name native_compare_std_step62 --seed 42 \
    --api_base "http://127.0.0.1:${PORT}/v1/" --api_key EMPTY \
    --model_id "${SERVED_NAME}" --max_tokens 8192 --max_retries 3 --parallel_workers 128 \
    > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/native_compare_infer_${BENCH}_$(date +%Y%m%d_%H%M%S).log 2>&1

  python3 judge_qwenlm.py --benchmark "${BENCH}" --model native_compare_std_step62 \
    --api_base "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl" \
    --api_key "${AZURE_GPT_API_KEY}" --api_type azure --api_version "2024-02-01" \
    --judge_model "gpt-5.4-mini-2026-03-17" --judge_max_tokens 2048 \
    > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/native_compare_judge_${BENCH}_$(date +%Y%m%d_%H%M%S).log 2>&1

  python3 cal_acc.py --benchmark "${BENCH}" --judge_json "judge/${BENCH}/native_compare_std_step62_answer.jsonl" \
    --benchmark_json "$PWD/${JSON}" \
    > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/native_compare_acc_${BENCH}_$(date +%Y%m%d_%H%M%S).log 2>&1
done

kill $SERVER_PID 2>/dev/null
echo "[$(date)] native eval 全部跑完，结果在 logs/native_compare_acc_*.log 里，对照 VLMEvalKit 已知分数(vstar 76.96/hrbench4k 77.62/hrbench8k 73.75)"
