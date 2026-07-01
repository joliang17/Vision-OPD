# SR1 BaseOPSD + RES vs Qwen3-VL-2B-Instruct Baseline

Saved: 2026-06-30

## Purpose

This snapshot records the current `sr1_ckpt*_baseopsd` results for comparing an added-RES training run against the original `Qwen3-VL-2B-Instruct` baseline across the same VLMEvalKit setting.

## Model Sources

Baseline:

`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit/outputs_vllm_curated/qwen3vl2b_base_reference_temp0_4096_eval/qwen3vl2b_base_reference_temp0_4096/T20260628-233419`

BaseOPSD + RES checkpoints:

`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/unsup-opsd/work_dirs/opsd_vlm/vision-sr1-47k/qwen3vl2b_answer_opsd_4gpu_s256_t1536/checkpoint-50-merged`

`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/unsup-opsd/work_dirs/opsd_vlm/vision-sr1-47k/qwen3vl2b_answer_opsd_4gpu_s256_t1536/checkpoint-100-merged`

`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/unsup-opsd/work_dirs/opsd_vlm/vision-sr1-47k/qwen3vl2b_answer_opsd_4gpu_s256_t1536/checkpoint-150-merged`

Eval output directories:

`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit/outputs_vllm_curated/sr1_ckpt50_baseopsd_qwen3vl2b_temp0_4096_generic_eval`

`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit/outputs_vllm_curated/sr1_ckpt100_baseopsd_qwen3vl2b_temp0_4096_generic_eval`

`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit/outputs_vllm_curated/sr1_ckpt150_baseopsd_qwen3vl2b_temp0_4096_generic_eval`

## Eval Setting

- Model family: `Qwen3-VL-2B-Instruct`
- Decoding: `temperature=0.0`, `max_new_tokens=4096`, `top_p=0.8`, `top_k=20`
- Penalties: `repetition_penalty=1.0`, `presence_penalty=1.5`
- Normal scoring judge: `gpt-5.4-mini-2026-03-17`
- Benchmarks: `BLINK`, `MMStar`, `MMBench_DEV_EN`, `VStarBench`, `MathVista_MINI`
- Unit: percent accuracy
- Delta: checkpoint score minus baseline score, in percentage points

## Summary

| Benchmark | Baseline | ckpt50 | Δ50 | ckpt100 | Δ100 | ckpt150 | Δ150 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLINK | 53.45 | 54.60 | +1.16 | 54.23 | +0.79 | 54.13 | +0.68 |
| MMStar | 55.00 | 57.07 | +2.07 | 57.67 | +2.67 | 57.60 | +2.60 |
| MMBench_DEV_EN | 77.92 | 77.15 | -0.77 | 77.58 | -0.34 | 76.46 | -1.46 |
| VStarBench | 72.77 | 72.25 | -0.52 | 72.77 | +0.00 | 73.30 | +0.52 |
| MathVista_MINI | 62.10 | 61.90 | -0.20 | 62.70 | +0.60 | 62.20 | +0.10 |

## Notes

- `ckpt100` is the strongest checkpoint in this snapshot by aggregate behavior: best `MMStar`, best `MathVista_MINI`, and smaller `MMBench_DEV_EN` drop than `ckpt150`.
- `ckpt150` has the best `VStarBench`, but the largest `MMBench_DEV_EN` drop.
- These are the `baseopsd` results, not the separate `pure-jsd-res` jobs that were started later.
