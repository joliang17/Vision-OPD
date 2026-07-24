#!/bin/bash
# 本机(301783374)专注跑training，不跑任何评测。
# 保守×sr1-90step(在跑,GPU0-3) 结束后 -> 只启动 qtext 训练(GPU0-3)，不接merge/eval。
# merge+评测的命令都记录在 docs/queued_experiments_20260713.md，交给别的机器跑。
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

echo "[$(date)] 等待保守×sr1-90step训练结束..."
while pgrep -f "trainer.experiment_name=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] 保守×sr1训练结束，启动qtext训练(GPU0-3)"

CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT=qvis EXPERIMENT_NAME=Vision-OPD-contrast-standard-qtext-Qwen3-VL-2B-Instruct \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  'actor_rollout_ref.actor.self_distillation.ra_generic_prompt=What is the answer?' \
  > logs/qtext_train_only_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!
wait $PID
echo "[$(date)] qtext训练结束。本机没有更多排队训练——检查是否需要新任务。不自动跑merge/eval。"
