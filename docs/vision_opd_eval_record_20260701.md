# Vision-OPD Qwen3-VL-2B Eval Record (2026-07-01)

This note records the main VLMEvalKit result locations and the currently consolidated scores for the Vision-OPD 2B variants stored on the shared NAS.

## Model Checkpoints

- `Vision-OPD-baseline-Qwen3-VL-2B-Instruct/global_step_65`
- `Vision-OPD-noimg-Qwen3-VL-2B-Instruct/global_step_62`
- `Vision-OPD-black-Qwen3-VL-2B-Instruct/global_step_65`
- `Vision-OPD-degrade-Qwen3-VL-2B-Instruct/global_step_62`
- `Vision-OPD-qvis-Qwen3-VL-2B-Instruct/global_step_62`
- `Vision-OPD-visionopd-Qwen3-VL-2B-Instruct/global_step_65`

## Training Setting And Loss

This section records the training setup used by the Vision-OPD 2B runs, so the method can be compared against related papers. The source of truth is the Vision-OPD training repo:

- launcher: `/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/scripts/run_vision_opd_ra_vad.sh`
- experiment wrappers: `/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/scripts/run_experiment_*.sh`
- self-distillation loss: `/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/verl/trainer/ppo/core_algos.py`
- RA-VAD weighting/loss: `/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/verl/trainer/ppo/ra_vad.py`
- actor update path: `/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/verl/workers/actor/dp_actor.py`

### Common Setup

- base model: `Qwen/Qwen3-VL-2B-Instruct`
- local model snapshot used by the runs:
  `/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`
- training data:
  - `data/train.parquet`: 6241 rows, columns include `prompt`, `images`, `bbox_images`, `reward_model`, `extra_info`
  - answer validation split enabled by default:
    - `data/train_answer.parquet`: 5985 rows
    - `data/val_answer.parquet`: 256 rows
  - degrade control additionally uses `data/train_degraded.parquet` / `data/train_degraded_answer.parquet`, with `images_degraded`
- optimizer/training scale:
  - `trainer.total_epochs=1`
  - `trainer.n_gpus_per_node=4`
  - `data.train_batch_size=96`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=96`
  - `actor_rollout_ref.actor.optim.lr=2e-6`
  - `actor_rollout_ref.actor.optim.lr_warmup_steps=10`
  - `actor_rollout_ref.actor.use_dynamic_bsz=True`
  - actor/ref parameter offload enabled; actor optimizer offload enabled
- rollout:
  - `actor_rollout_ref.rollout.name=vllm`
  - `actor_rollout_ref.rollout.n=8`
  - `actor_rollout_ref.rollout.response_length=1024`
  - `data.max_prompt_length=8192`
  - `data.max_response_length=1024`
  - `actor_rollout_ref.rollout.max_model_len=9216`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=9216`
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.7`
- training objective switch:
  - `actor_rollout_ref.actor.policy_loss.loss_mode=vopd`
  - `algorithm.adv_estimator=grpo`, but the actor update bypasses the normal GRPO policy-gradient loss for samples covered by VOPD/self-distillation. GRPO only appears as fallback when a sample has no usable teacher target.
  - `actor_rollout_ref.actor.use_kl_loss=False`
  - `algorithm.use_kl_in_reward=False`
  - `reward_model.enable=False`
- teacher:
  - `teacher_model_source=legacy`
  - `teacher_regularization=ema`
  - `teacher_update_rate=0.05`
  - `teacher_always_on=True`
  - `dont_reprompt_on_self_success=True`
  - `include_environment_feedback=False`

### Variant Definitions

| row name | experiment | teacher high-information input | control input | RA-VAD | distillation logits |
| --- | --- | --- | --- | --- | --- |
| `visionopd` | `visionopd` | bbox crop from `bbox_images` | none | false | top-k, `distillation_topk=100` |
| `baseline` | `baseline` | original image from `images` plus ground-truth answer hint | none | false | top-k, `distillation_topk=100` |
| `degrade` | `degrade` | original image from `images` | degraded image from `images_degraded` | true | full vocabulary |
| `qvis` | `qvis` | original image from `images` | original image with generic prompt `Describe this image in detail.` | true | full vocabulary |
| `noimg` | `noimg` | original image from `images` | text-only / image stripped | true | full vocabulary |
| `black` | `black` | original image from `images` | same-size solid-black image | true | full vocabulary |

The `baseline` answer-hint teacher prompt appends the reference answer to the original problem:

```text
Here is a reference solution to this problem:
{answer}

After understanding the reference solution, please try to solve this problem using your own approach below:
```

### Loss Details

For non-RA VOPD variants (`visionopd`, `baseline`), the actor minimizes a self-distillation KL between the student distribution and the teacher distribution on response tokens.

- `full_logit_distillation=True`
- `alpha=0.5`
- `is_clip=2.0`
- `distillation_topk=100`
- `distillation_add_tail=True`
- loss aggregation: `token-mean`

In code, `compute_self_distillation_loss` supports forward KL (`alpha=0`), reverse KL (`alpha=1`), and generalized Jensen-Shannon divergence for intermediate `alpha`. These runs use `alpha=0.5`, so the top-k distillation objective is the generalized JSD between student and teacher distributions, with an importance-ratio clip from old policy log-probs capped at `2.0`.

For RA-VAD variants (`degrade`, `qvis`, `noimg`, `black`), the actor first computes response-token relevance weights from the teacher likelihood gap between high-information input and control input:

```text
ra_raw_t = log p_teacher(y_t | high-info input, y_<t)
           - log p_teacher(y_t | control input, y_<t)
ra_pos_t = max(ra_raw_t - delta, 0)
```

Then positive weights are clipped by quantile, normalized per sample, optionally gated at sample level, and used as stop-gradient token weights for full-vocabulary teacher-student KL:

```text
L_RA-VAD = sum_t w_t * KL(p_teacher(. | high-info input) || p_student(. | original input))
           / sum_t w_t
```

RA-VAD hyperparameters in these runs:

- `ra_delta=0.0`
- `ra_clip_quantile=0.95`
- `ra_min_positive_tokens=1`
- `ra_temperature=2.0`
- `ra_uniform_weight=False`
- `ra_no_sample_gate=False`
- `ra_margin_scale=0.5`
- `ra_answer_scale=0.1`
- `distillation_topk=null`
- `full_logit_distillation=True`

### RA-VAD Weight Interpretation

The current RA-VAD implementation only keeps positive teacher likelihood gaps:

```text
ra_raw_t = logp_hi_t - logp_ctrl_t
ra_pos_t = max(ra_raw_t - delta, 0)
```

So a token receives nonzero RA weight only when the high-information teacher condition assigns higher likelihood than the control condition. Negative gaps are ignored by the current loss.

The relation between `ra_pos_t` and final token weight `w_t` is:

```text
positive_mask_t = 1[ra_pos_t > 0]
ra_pos_t = min(ra_pos_t, sample_quantile(ra_pos, q=0.95))
ra_norm_t = ra_pos_t / mean_positive(ra_pos)
w_t = stop_grad(ra_norm_t * g_sample)
```

The quantile clipping is per sample. It caps extreme positive `ra_pos_t` values at the 95th percentile among positive tokens in the same response. This keeps the token-selection signal but prevents one or two very large likelihood gaps from dominating the loss.

The sample-level gate is:

```text
margin = mean_t(ra_raw_t)
ra_answer = mean_t(ra_norm_t over positive tokens)
g_sample = sigmoid(margin / ra_margin_scale)
           * sigmoid(ra_answer / ra_answer_scale)
```

`ra_norm_t` decides which tokens inside a response matter; `g_sample` decides how strongly the whole response should be used. If the high-information condition is not better than the control condition on average, `margin` is small or negative and the whole sample is downweighted.

### Negative-Token Design Question

Negative gaps are meaningful but are not used by the current method:

```text
ra_raw_t < 0  <=>  logp_ctrl_t > logp_hi_t
```

This can happen for several reasons:

- the control input may actually correct or avoid misleading visual evidence from the full/high-information input;
- the high-information input may introduce distracting objects, visual clutter, or wrong fine details;
- the token may be mostly language/format driven, where the control condition has a stronger prior;
- the teacher may be unstable across prompt/image conditions, so the likelihood gap is only a proxy for correctness.

The current design treats positive gaps as evidence-sensitive tokens and drops negative gaps. This is conservative: it avoids distilling from a control condition that may be driven by language priors, prompt artifacts, or accidental correctness.

A possible signed/symmetric variant to discuss would be:

```text
w_pos_t = ReLU(logp_hi_t - logp_ctrl_t)
w_neg_t = ReLU(logp_ctrl_t - logp_hi_t)
```

Potential uses of `w_neg_t`:

- use it as a filter or uncertainty signal to reduce weight on samples/tokens where high-info is worse than control;
- distill from the control teacher on negative tokens;
- compare high-info and control answers and only keep negative tokens when the control answer is independently judged correct;
- use negative mass as a diagnostic metric rather than a training signal.

The risk of training on negative tokens is that it changes the method from "distill visual-evidence-sensitive tokens from the high-information condition" to "choose between two teacher conditions." Without an additional correctness signal, negative tokens may inject language priors or control-condition artifacts into the student.

For `degrade`, the degraded control image budget is configured with:

- `ra_ctrl_low_tokens=256`
- `ra_ctrl_up_tokens=1536`

### Method-Comparison Notes

- This is not plain supervised fine-tuning on final answers: the training loss is teacher-student distribution matching over generated response tokens.
- This is not standard RLHF/GRPO as the main objective for these checkpoints: `loss_mode=vopd` replaces the usual policy-gradient loss when a teacher target is available, and reward/KL-in-reward are disabled for the training objective.
- The main paper-overlap-sensitive component is the RA-VAD weighting: it uses a paired high-information vs control teacher likelihood gap to select/weight visual-evidence-sensitive response tokens, then distills the full teacher distribution on those tokens.
- The control-input ablations are methodologically important:
  - `noimg`: isolates visual contribution against text-only control.
  - `black`: isolates visual contribution against a same-size non-informative visual control.
  - `degrade`: isolates high-resolution/detail contribution against a degraded image.
  - `qvis`: isolates answer-task visual evidence against generic image-description prompting.

## Main Result Directories

### Baseline

- `outputs_vllm_curated/visionopd_qwen3vl2b_step65_temp0_4096_plus_hrbench_0701_qwen3vl2b_temp0_4096_generic_eval`

### Noimg

- `outputs_vllm_curated/visionopd_noimg_qwen3vl2b_step62_temp0_4096_plus_hrbench_0701_qwen3vl2b_temp0_4096_generic_eval`

### Black

- `outputs_vllm_curated/visionopd_2B_blackckp65_qwen3vl2b_temp0_4096_generic_eval`

### Degrade

- `outputs_vllm_curated/visionopd_2B_resckp62_qwen3vl2b_temp0_4096_generic_eval`

### Qvis

- `outputs_vllm_curated/visionopd_2B_qvisckp62_qwen3vl2b_temp0_4096_generic_eval`

### VisionOPD

- Main root:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038`
- Core strict scoring:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_core_strict_rejudge_0630`
- HRBench4K strict rerun/scoring:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_hr4k_strict_rejudge_0701`
- HRBench8K strict scoring:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/hrbench_rejudge_gpt54_wait_0630`
- Raw HR/Zoom inference:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_hr_zoom_eval`
- HRBench4K rerun inference:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_hr4k_rerun_gpu2`

## Consolidated Scores

Note: the `visionopd` row below comes from the strict/rejudge path above. The other rows come from `normal_scoring`.

| model | BLINK | MMStar | MMBench_DEV_EN | VStarBench | MathVista_MINI | HRBench4K | HRBench8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 55.44 | 58.87 | 0.6924 | 72.25 | 61.8 | 0.7325 | 0.6963 |
| noimg | 55.50 | 61.20 | 0.7655 | 73.82 | 63.2 | 0.7575 | 0.7138 |
| black | 54.71 | 60.33 | 0.7646 | 73.82 | 63.6 | 0.7525 | 0.7063 |
| degrade | 54.45 | 61.87 | 0.7509 | 74.35 | 64.9 | 0.7513 | 0.7175 |
| qvis | 53.34 | 61.20 | 0.7543 | 71.20 | 62.7 | 0.7525 | 0.7125 |
| visionopd | 47.55 | 58.60 | 0.7552 | 75.39 | 61.1 | 0.7488 | 0.7038 |

## Source Files Used For VisionOPD Row

- BLINK:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_core_strict_rejudge_0630/preds/visionopd_qwen3vl2b_step65_temp0_4096_correct_BLINK_strict_acc.csv`
- MMStar:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_core_strict_rejudge_0630/preds/visionopd_qwen3vl2b_step65_temp0_4096_correct_MMStar_strict_acc.csv`
- MMBench_DEV_EN:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_core_strict_rejudge_0630/preds/visionopd_qwen3vl2b_step65_temp0_4096_correct_MMBench_DEV_EN_strict_acc.csv`
- VStarBench:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_core_strict_rejudge_0630/preds/visionopd_qwen3vl2b_step65_temp0_4096_correct_VStarBench_strict_acc.csv`
- MathVista_MINI:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_core_strict_rejudge_0630/preds/visionopd_qwen3vl2b_step65_temp0_4096_correct_MathVista_MINI_strict_gpt-5.4-mini-2026-03-17_score.csv`
- HRBench4K:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/visionopd_hr4k_strict_rejudge_0701/preds/visionopd_qwen3vl2b_step65_temp0_4096_hr_judge_HRBench4K_normal_acc.csv`
- HRBench8K:
  `outputs_vllm_curated/remaining_vlmeval_20260630_2038/hrbench_rejudge_gpt54_wait_0630/preds/visionopd_qwen3vl2b_step65_temp0_4096_hr_judge_HRBench8K_normal_acc.csv`

## Targeted Reruns

The following targeted reruns were launched on 2026-07-01 using the current default eval setting:

- inference: `temperature=0.0`, `max_new_tokens=4096`
- judge: `gpt-5.4-mini-2026-03-17`, provider `tiktok_azure`, `temperature=0.0`
- scoring mode: `normal_scoring`

| target | previous score | rerun score | delta | judge API failures | judge parse failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `visionopd` / `BLINK` | 47.5539 | 47.5013 | -0.0526 | 0 | 0 |
| `baseline` / `MMBench_DEV_EN` | 0.6924 | 0.6942 | +0.0017 | 0 | 0 |
| `Qwen3-VL-2B-Instruct` / `HRBench4K` | 0.71125 | 0.71125 | 0 | 0 | 0 |

### Rerun Output Directories

- `visionopd` / `BLINK`:
  `outputs_vllm_curated/visionopd_step65_blink_rerun_20260701_qwen3vl2b_temp0_4096_generic_eval/normal_scoring`
- `baseline` / `MMBench_DEV_EN`:
  `outputs_vllm_curated/baseline_step65_mmbench_rerun_20260701_qwen3vl2b_temp0_4096_generic_eval/normal_scoring`
- `Qwen3-VL-2B-Instruct` / `HRBench4K`:
  `outputs_vllm_curated/qwen3vl2b_instruct_hrbench4k_rerun_20260701_qwen3vl2b_temp0_4096_generic_eval/normal_scoring`

### Rerun Score Files

- `visionopd` / `BLINK`:
  `outputs_vllm_curated/visionopd_step65_blink_rerun_20260701_qwen3vl2b_temp0_4096_generic_eval/normal_scoring/visionopd_step65_blink_rerun_20260701_BLINK_normal_acc.csv`
- `baseline` / `MMBench_DEV_EN`:
  `outputs_vllm_curated/baseline_step65_mmbench_rerun_20260701_qwen3vl2b_temp0_4096_generic_eval/normal_scoring/baseline_step65_mmbench_rerun_20260701_MMBench_DEV_EN_normal_acc.csv`
- `Qwen3-VL-2B-Instruct` / `HRBench4K`:
  `outputs_vllm_curated/qwen3vl2b_instruct_hrbench4k_rerun_20260701_qwen3vl2b_temp0_4096_generic_eval/normal_scoring/qwen3vl2b_instruct_hrbench4k_rerun_20260701_HRBench4K_normal_acc.csv`
