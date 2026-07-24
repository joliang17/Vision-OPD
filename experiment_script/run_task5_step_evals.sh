#!/bin/bash
# 等 task5 四个90步实验都跑完后，merge+评测 step30/60/90 三个checkpoint，用 BACKEND=vllm_server
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

EXPS=(
  "Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step:std_virl39k"
  "Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step:cons_virl39k"
  "Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered-90step:std_sr1"
  "Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step:cons_sr1"
)
STEPS=(30 60 90)

# 1. 等所有相关训练进程结束（按 EXPERIMENT_NAME 找 main_ppo 进程）
echo "[$(date)] 等待 task5 四个实验的训练进程全部结束..."
while true; do
  still_running=0
  for pair in "${EXPS[@]}"; do
    name="${pair%%:*}"
    if pgrep -f "trainer.experiment_name=${name} " >/dev/null 2>&1 || pgrep -f "trainer.experiment_name=${name}$" >/dev/null 2>&1; then
      still_running=1
    fi
  done
  if [ "$still_running" -eq 0 ]; then
    break
  fi
  sleep 30
done
echo "[$(date)] 全部训练进程已结束，开始 merge"

# 2. merge 每个实验的 step30/60/90（如果还没merge过，用 actor/huggingface 目录是否存在判断）
for pair in "${EXPS[@]}"; do
  name="${pair%%:*}"
  for step in "${STEPS[@]}"; do
    ckpt_dir="checkpoints/${name}/global_step_${step}"
    if [ -d "${ckpt_dir}/actor" ]; then
      if [ ! -f "${ckpt_dir}/config.json" ]; then
        bash scripts/merge_checkpoint.sh "${ckpt_dir}" > "logs/merge_${name}_step${step}_$(date +%Y%m%d_%H%M%S).log" 2>&1
      fi
    else
      echo "[WARN] ${ckpt_dir}/actor 不存在，跳过merge（可能该实验没跑到这一步）" >> logs/task5_step_evals_driver.log
    fi
  done
done
echo "[$(date)] merge 完成，开始评测"

# 3. 评测：4实验 x 3 checkpoint = 最多12个，每个用1张卡，4个一批（GPU0-3）跑完再下一批
cd VLMEvalKit
JOBS=()
for pair in "${EXPS[@]}"; do
  name="${pair%%:*}"
  tag="${pair##*:}"
  for step in "${STEPS[@]}"; do
    # std_virl39k 的 step30/60 由 run_std_virl39k_3060_eval_asap.sh 单独抢空卡跑了，这里跳过避免重复
    if [ "${tag}" == "std_virl39k" ] && { [ "${step}" == "30" ] || [ "${step}" == "60" ]; }; then
      continue
    fi
    ckpt_dir="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/${name}/global_step_${step}"
    if [ -f "${ckpt_dir}/config.json" ]; then
      JOBS+=("${ckpt_dir}|${tag}_step${step}")
    fi
  done
done

i=0
n=${#JOBS[@]}
while [ $i -lt $n ]; do
  pids=()
  for gpu in 0 1 2 3; do
    if [ $i -ge $n ]; then break; fi
    job="${JOBS[$i]}"
    ckpt_dir="${job%%|*}"
    tag="${job##*|}"
    BACKEND=vllm_server \
    MODEL_PATH="${ckpt_dir}" \
    MODEL_NAME="task5_${tag}_server" \
    DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
    GPU_IDS=${gpu} \
    nohup bash shell_scripts/eval_model_temp0_4096.sh \
      > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_task5_${tag}_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    pids+=("$!")
    i=$((i+1))
  done
  for p in "${pids[@]}"; do
    wait "$p"
  done
done

echo "[$(date)] task5 全部 step30/60/90 评测完成"
