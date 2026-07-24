#!/bin/bash
# 2026-07-15 大重启：前一夜下游链条三重故障(tensorboard缺失/termcolor缺失/端口8010撞车)后的统一补跑
# 根因已修：pip install tensorboard termcolor num2words hf_transfer litellm socksio openpyxl;
#          uniform-weight checkpoint tokenizer已从同基座覆盖(坑#7);
#          所有 vLLM 端口全局唯一(8041-8045)
# 已真实完成无需重跑：全部ZoomBench(除std30被撞车的26.27%)、native compare的vstar(80.10)/hrbench-4k(78.38)
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

ninebench() {  # <ckpt> <tag> <gpu>
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server MODEL_PATH="$1" MODEL_NAME="$2" \
  DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
  GPU_IDS=$3 bash shell_scripts/eval_model_temp0_4096.sh \
    > "$V/logs/r2_eval_$2_$(date +%Y%m%d_%H%M%S).log" 2>&1
  cd "$V"
}

lane_gpu03() {
  # 训练1: 保守×sr1-90step (从0, 4096)
  CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step \
    ANSWER_VAL_TRAIN_FILE=$V/data/vision_sr1_47k_noimg_v2_filtered.parquet \
    TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
    bash scripts/run_experiment_contrast_conservative.sh \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/r2_train_cons_sr1_$(date +%Y%m%d_%H%M%S).log 2>&1
  # 训练2: qtext
  CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT=qvis EXPERIMENT_NAME=Vision-OPD-contrast-standard-qtext-Qwen3-VL-2B-Instruct \
    bash scripts/run_experiment_contrast_standard.sh \
    'actor_rollout_ref.actor.self_distillation.ra_generic_prompt=What is the answer?' \
    > logs/r2_train_qtext_$(date +%Y%m%d_%H%M%S).log 2>&1
  # merge + 评测（GPU0: cons_sr1三个ckpt; GPU1: qtext）并行
  (
    NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step
    for s in 30 60 90; do
      d=checkpoints/$NAME/global_step_$s
      [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" > logs/r2_merge_cons_sr1_$s.log 2>&1
      [ -f "$d/config.json" ] && ninebench "$V/$d" "cons_sr1_90step_step$s" 0
      [ -f "$d/config.json" ] && bash scripts/run_zoombench_canonical.sh "$d" "cons_sr1_90step_step$s" 0 $((8050+s)) > logs/r2_zoombench_cons_sr1_$s.log 2>&1
    done
  ) &
  (
    NAME=Vision-OPD-contrast-standard-qtext-Qwen3-VL-2B-Instruct
    FS=$(cat checkpoints/$NAME/latest_checkpointed_iteration.txt 2>/dev/null)
    if [ -n "$FS" ]; then
      d=checkpoints/$NAME/global_step_$FS
      [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" > logs/r2_merge_qtext.log 2>&1
      ninebench "$V/$d" "contrast_standard_qtext_step$FS" 1
      bash scripts/run_zoombench_canonical.sh "$d" "contrast_standard_qtext_step$FS" 1 8042 > logs/r2_zoombench_qtext.log 2>&1
    fi
  ) &
  wait
  echo "[$(date)] lane_gpu03 完成"
}

lane_gpu4() {
  # 第三组对照: VLMEvalKit×原生配置(8192+pp0)
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server \
  MODEL_PATH=$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct/global_step_62 \
  MODEL_NAME=native_setting_std_step62_8192_pp0 \
  DATASETS=VStarBench,HRBench4K,HRBench8K MAX_NEW_TOKENS=8192 PRESENCE_PENALTY=0 GPU_IDS=4 \
  bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/r2_native_setting_eval_$(date +%Y%m%d_%H%M%S).log" 2>&1
  cd "$V"
  # native compare 的 hrbench-8k 重推理(被端口撞车毁掉的那个), 独占端口8043
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
  source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
  CKPT=$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct/global_step_62
  CUDA_VISIBLE_DEVICES=4 nohup vllm serve "$CKPT" --served-model-name native_compare_std_step62 \
    --host 0.0.0.0 --port 8043 --trust-remote-code --enforce-eager --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --dtype bfloat16 --limit-mm-per-prompt '{"image":8}' \
    > logs/r2_native8k_serve.log 2>&1 &
  SP=$!
  until curl -s "http://127.0.0.1:8043/v1/models" >/dev/null 2>&1; do
    sleep 10; kill -0 $SP 2>/dev/null || { echo "serve died"; return 1; }
  done
  cd eval
  rm -f model_answer/hrbench-8k/native_compare_std_step62_answer.jsonl judge/hrbench-8k/native_compare_std_step62_answer.jsonl
  python3 infer.py --benchmark hrbench-8k --benchmark_json "$PWD/hr_bench_8k.json" \
    --out_dir model_answer --model_name native_compare_std_step62 --seed 42 \
    --api_base "http://127.0.0.1:8043/v1/" --api_key EMPTY --model_id native_compare_std_step62 \
    --max_tokens 8192 --max_retries 3 --parallel_workers 128 > "$V/logs/r2_native8k_infer.log" 2>&1
  kill $SP 2>/dev/null
  python3 judge_qwenlm.py --benchmark hrbench-8k --model native_compare_std_step62 \
    --api_base "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl" \
    --api_key "${AZURE_GPT_API_KEY}" --api_type azure --api_version "2024-02-01" \
    --judge_model "gpt-5.4-mini-2026-03-17" --judge_max_tokens 2048 > "$V/logs/r2_native8k_judge.log" 2>&1
  python3 cal_acc.py --benchmark hrbench-8k --judge_json "judge/hrbench-8k/native_compare_std_step62_answer.jsonl" \
    --benchmark_json "$PWD/hr_bench_8k.json" > "$V/logs/r2_native8k_acc.log" 2>&1
  cd "$V"
  # std_sr1 30/60/90 的9bench(从GPU7泳道匀过来平衡时长)
  for s in 30 60 90; do
    ninebench "$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered-90step/global_step_$s" "std_sr1_90step_step$s" 4
  done
  echo "[$(date)] lane_gpu4 完成"
}

lane_gpu5() {
  # API失败重跑链(uniform tokenizer已修)
  ninebench_ds() { # <ckpt> <tag> <ds>
    cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
    BACKEND=vllm_server MODEL_PATH="$1" MODEL_NAME="$2" DATASETS="$3" GPU_IDS=5 \
    bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/r2_rerun_$2_$(date +%Y%m%d_%H%M%S).log" 2>&1
    cd "$V"
  }
  ninebench_ds $V/checkpoints/Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62 uniform_weight_hrbench8k_rerun HRBench8K
  ninebench_ds $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-Instruct/global_step_62 std4b_hrbench_rerun HRBench4K,HRBench8K
  ninebench_ds $V/checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-4B-Instruct/global_step_62 cons4b_hrbench_rerun HRBench4K,HRBench8K
  ninebench_ds $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered/global_step_140 std_sr1_step140_hrbench8k_rerun HRBench8K
  # uniform-weight 的 ZoomBench(tokenizer修好后), 端口8045
  bash scripts/run_zoombench_canonical.sh \
    checkpoints/Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62 \
    contrast_standard_uniform_weight_step62 5 8045 > logs/r2_zoombench_uniform.log 2>&1
  echo "[$(date)] lane_gpu5 完成"
}

lane_gpu6() {
  ninebench $V/checkpoints/Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered-3ep/global_step_1392 grpo_virl39k_3ep_step1392 6
  ninebench $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30 std_virl39k_90step_step30 6
  ninebench $V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_60 std_virl39k_90step_step60 6
  # std30 的 ZoomBench 重跑(26.27%疑似被端口撞车污染), 端口8044
  bash scripts/run_zoombench_canonical.sh \
    checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30 \
    contrast_standard_virl39k_90step_step30_rerun 6 8044 > logs/r2_zoombench_std30_rerun.log 2>&1
  echo "[$(date)] lane_gpu6 完成"
}

lane_gpu7() {
  for s in 30 60 90; do
    ninebench $V/checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_$s cons_virl39k_90step_step$s 7
  done
  echo "[$(date)] lane_gpu7 完成"
}

lane_gpu03 & P1=$!
lane_gpu4 & P2=$!
lane_gpu5 & P3=$!
lane_gpu6 & P4=$!
lane_gpu7 & P5=$!
wait $P1 $P2 $P3 $P4 $P5
echo "[$(date)] 全部泳道完成。下一步:scan_eval_integrity.py 重扫 + 回填文档"
