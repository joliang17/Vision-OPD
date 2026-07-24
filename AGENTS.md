# Repository Guidelines

## Project Structure & Module Organization

Vision-OPD is a Python research codebase built around the `verl` package. Core training, rollout, worker, model-merging, and utility code lives under `verl/`. Entry-point scripts are in `scripts/`, including data preparation, training variants, checkpoint merging, and experiment reruns. Evaluation code is in `eval/`, with benchmark assets and generated answers stored below that directory. `data/` holds training data and images, `checkpoints/`, `rollouts/`, `logs/`, `outputs/`, and `tensorboard_log/` are runtime artifacts, and `external/VLMEvalKit/` is a vendored evaluation dependency.

## Build, Test, and Development Commands

Set up the documented environment from the repository root:

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
```

Prepare data with `python scripts/prepare_data.py --data-dir ./data`. Start the main training flow with `bash scripts/run_vision_opd.sh`. Merge FSDP checkpoints with `bash scripts/merge_checkpoint.sh <checkpoint_dir>`. Run benchmark evaluation through `eval/run_eval.sh`, setting `API_BASE`, `OPENAI_MODEL_ID`, `JUDGE_API_BASE`, `JUDGE_MODEL`, and `BENCHMARK` as needed.

### GRPO Baseline Commands

Use this command for the vanilla GRPO baseline on the Vision-OPD training data. This keeps the model, data, chat template, LR, rollout size, response length, save frequency, and GPU count comparable to OPD runs, while using `loss_mode=vanilla` and an accuracy-only reward.

```bash
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
EXPERIMENT_NAME=grpo_vanilla_qwen3vl2b_3ep \
TRAINER_N_GPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=32 \
PPO_MINI_BATCH_SIZE=8 \
ROLLOUT_N=8 \
LR=1e-6 \
MAX_PROMPT_LENGTH=8192 \
MAX_RESPONSE_LENGTH=1024 \
TRAINER_TOTAL_EPOCHS=3 \
TRAINER_SAVE_FREQ=20 \
FORMAT_REWARD_WEIGHT=0.0 \
bash scripts/run_experiment_grpo_baseline.sh
```

For the matched OPD answer-hint comparison, use the same command and append the OPD-only overrides:

```bash
  actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
  actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
  actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=answer_hint
```

For a 5 epoch run, change `TRAINER_TOTAL_EPOCHS=3` to `TRAINER_TOTAL_EPOCHS=5` and update `EXPERIMENT_NAME` from `_3ep` to `_5ep`.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax; the README environment uses Python 3.12. Follow existing package naming: modules and functions use `snake_case`, classes use `PascalCase`, and experiment shell scripts use descriptive lower-case names such as `run_experiment_visionopd.sh`. Ruff is configured in `pyproject.toml` with a 120-character line length, import sorting, and checks for `E`, `F`, `UP`, `B`, `I`, and `G`. Run `ruff check .` before submitting Python changes when Ruff is installed.

## Testing Guidelines

This repository does not currently expose a top-level test suite. For Python logic, add focused `test_*.py` files near the touched component or under a future top-level `tests/` directory, and prefer small fixtures over large model or image artifacts. For evaluation changes, validate with a narrow `BENCHMARK` selection before launching full runs. Vendored tests live in `external/VLMEvalKit/tests/` and should be treated as dependency coverage, not primary project tests.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Fix VisionOPD strict judge probe` and `Add Vision-OPD eval and rerun utilities`. Keep commits scoped to one behavior or experiment path. Pull requests should describe the motivation, list changed commands or configs, note required model/data paths, and include representative logs, metrics, or benchmark names for training and evaluation changes.

## Security & Configuration Tips

Do not commit credentials, judge API keys, downloaded datasets, checkpoints, or generated rollouts. Pass service endpoints and model identifiers through environment variables, as shown in `eval/run_eval.sh`.
