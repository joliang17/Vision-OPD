# Vision-OPD — Claude Context

## What This Repo Is

Vision-OPD is a reinforcement-learning fine-tuning framework for vision-language models. Training uses a GRPO/OPD loop (`verl/` package). Checkpoints land in `checkpoints/`. Evaluation lives entirely in `eval/`.

## Checkpoint Retention (standing policy, 2026-07-10)

The bytefuse volume has a shared per-user quota, and `checkpoints/` hitting it **crashed two live
trainings on 2026-07-10** (`Disk quota exceeded` writing rollout dumps — the volume itself, not
wandb). Each 2B checkpoint dir is ~27GB (mostly optimizer shards). Standing policy from the user:

- **Keep dense saving ON during training** (`trainer.save_freq=10` — it's what makes crash
  resume cheap; that same crash lost <1 step of progress thanks to it).
- **After a run completes, immediately prune**: ~60-step runs keep **3** checkpoints total,
  400+-step runs keep **5**, evenly spaced, always including the final step (plus any step that
  already has published eval numbers). Use `bash scripts/prune_checkpoints.sh <run_dir> <n>`.
- Never prune another session's run directory — check `ps`/recent mtimes first.

## Environment Gotchas

**⛔ NEVER kill/delete the `keep_gpu` process** (`keep_gpu.mcp.server`, holds ~1748MiB per GPU and fakes
utilization spikes): the platform reclaims (kills) pods whose GPU utilization stays low — keep_gpu is the
occupancy keepalive. **A machine in this project was killed on 2026-07-16 immediately after its keep_gpu was
deleted.** When cleaning up training/eval processes, never use broad kill patterns (`pkill -f python`); match
specific names only (`TaskRunner|ray::|vllm serve|verl.trainer.main_ppo`). The 1748MiB-per-GPU baseline in
`nvidia-smi` and the fake 100% utilization ARE keep_gpu — leave them alone; "idle" on these boxes means memory
at that baseline, not zero. If the baseline ever disappears, keep_gpu died — alert the user, the pod is at
reclamation risk.

**⚠️ Liveness checks on verl trainings MUST scan `/proc/<pid>/cmdline`**: verl's `main_ppo` command line
is ~8KB, and BOTH `pgrep -f` AND `ps aux` truncate it (~4KB) — `EXPERIMENT_NAME`/checkpoint-dir strings sit
past the cut, so pattern matches silently fail. This caused three incidents on 2026-07-17 alone (a driver
waited 9h on phantom processes; a supervisor false-reported a live run dead; a batch driver launched new
trainings onto GPUs still occupied by running jobs). Correct pattern:
```bash
proc_alive() { local p; for p in $(pgrep -f main_ppo); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "$1" && return 0; done; return 1; }
```
Pair it with a target-GPU memory check (`nvidia-smi ... < 10GB`) before launching anything.

**⚠️ After a training OOM crash, sweep zombie `VLLM::EngineCore` processes before retrying**: when
verl's update_actor OOMs, the ray actors die but the vLLM rollout engine subprocess (shows as
`VLLM::EngineCore` in /proc, invisible to `pgrep -f main_ppo`) can survive holding its FULL memory pool
(~175GB observed, 2026-07-18) on its GPUs. Every retry then OOMs harder while pool-size levers appear
"not to work". Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i <gpu>` for
non-keep_gpu residents and kill them before any resume. (keep_gpu's ~1.7GB per GPU is NOT a zombie —
never kill it.)

**Proxy**: The machine has `http_proxy`/`https_proxy` set globally to an unreachable address, **and separately `ALL_PROXY`/`all_proxy` set to an unreachable SOCKS proxy** (`socks5h://127.0.0.1:1080`) — unsetting only the http(s)_proxy pair is not enough. `ALL_PROXY` alone is enough to break `litellm`/`httpx`-based Azure judge calls in VLMEvalKit with `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`, which silently falls back to VLMEvalKit's own exact-match grading instead of GPT-based judge matching — this contaminated several MCQ benchmark scores (BLINK/MMStar/MMBench_DEV_EN) on 2026-07-08 before being found. Always unset all four before running inference, vLLM, or anything that calls a judge:
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
```
This fix is baked into `VLMEvalKit/shell_scripts/eval_merge_infer_normal_score.sh` itself now, but still unset manually before any ad-hoc `python3 -c "..."` judge test or direct `tools/run_normal_eval.py` invocation.

**User-local site-packages can silently shadow the pinned system packages**: `~/.local/lib/python3.11/site-packages` takes precedence over `/usr/local/lib/python3.11/dist-packages`. An external provisioning/reset event on 2026-07-08 dropped a large batch of freshly-`pip install --user`'d packages there (numpy 2.2.6, pandas 3.0.3, transformers 5.12.1, huggingface_hub 1.22.0, antlr4-python3-runtime 4.11.1, and dozens more), which don't match what the pinned system install (numpy 1.26.4, pandas 1.5.3, transformers 4.57.0) was built/tested against — causing `ValueError: numpy.dtype size changed`, hydra ATN deserialization errors, and other cascading import failures. If you hit an import error that doesn't match anything in this file, check `ls -la --time-style=full-iso ~/.local/lib/python3.11/site-packages/ | sort -k6,7` for a suspicious batch of same-timestamp installs and `rm -rf` just those directories/dist-info folders (never the whole `.local` — other things may legitimately live there) to fall back to the system versions.

**prometheus_fastapi_instrumentator bug**: a bare `pip install -r requirements.txt` in `VLMEvalKit/` pulls `fastapi>=0.115.0` unconstrained, which drags in a `starlette` new enough that vLLM 0.11.0's `--served-model-name ... vllm serve` API server crashes on every request with `AttributeError: '_IncludedRouter' object has no attribute 'path'` (inside `prometheus_fastapi_instrumentator`'s route-name middleware). Fix is now pinned in `VLMEvalKit/requirements_arnold.txt`: `fastapi==0.115.6` / `starlette==0.41.3`. If a fresh/reset environment shows this crash on the first request to a `vllm serve` instance (health checks still pass — only real requests 500), re-run `pip install -r requirements_arnold.txt` in `VLMEvalKit/`, or manually `pip install fastapi==0.115.6 "starlette<1.0.0,>=0.40.0"`. (An earlier fix patched `prometheus_fastapi_instrumentator/routing.py` directly in site-packages instead — that patch does not survive an environment reset and the file may no longer exist; prefer the requirements pin.)

**GPUs**: 4× B200 (183 GB each). GPU 0 is reliably free. GPUs 1–3 are often occupied by other users. Always check with `nvidia-smi` before picking a GPU.

**Models**: HuggingFace is not reachable (no internet / proxy broken). Pass the full local cache path to vLLM, never the HF repo name. Pre-cached models:
- Qwen3-VL-2B: `/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`
- Qwen3-VL-4B: find with `find /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache -name "config.json" | xargs grep -l "Qwen3-VL-4B"`

## Eval Pipeline

**Default policy (updated 2026-07-08)**: use VLMEvalKit for every benchmark except ZoomBench — ZoomBench is not a standard VLMEvalKit dataset. **For ZoomBench, the default is now the native eval pipeline (`infer.py` → `judge_qwenlm.py` → `cal_acc.py`) with the three judge-bug fixes applied** — see "ZoomBench — canonical eval (default)" below. The older `run_official_eval.sh` (thinking-mode) pipeline is demoted to a historical/alternate reference, not the default — see the note in that section for why.

### Default multi-benchmark eval (use this) — VLMEvalKit

**Script**: `VLMEvalKit/shell_scripts/eval_model_temp0_4096.sh` (temp=0, max_new_tokens=4096)

```bash
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit

MODEL_PATH=<merged checkpoint dir, or CKPT_PATH= for a raw base/HF model dir> \
MODEL_NAME=<tag for output dir/filenames> \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,MME-RealWorld \
GPU_IDS=<free gpu> \
BACKGROUND=1 \
bash shell_scripts/eval_model_temp0_4096.sh
```

- FSDP/verl checkpoints must be merged first: `bash scripts/merge_checkpoint.sh <checkpoint dir>/global_step_N` (from `Vision-OPD/`) before pointing `MODEL_PATH` at it — this writes `config.json` etc. directly into that checkpoint dir.
- Judge defaults to Azure `gpt-5.4-mini-2026-03-17` via `tiktok_azure` provider, reading `opsd/config/key.conf` — no extra flags needed.
- `BACKGROUND=1` daemonizes immediately; the driver process exits right away by design — track the forked `pid=` and the `log=` path printed at startup, not the driver PID, when checking on progress.
- `MME-RealWorld` (English) is a valid VLMEvalKit dataset name and works through this same script — don't use the native `eval/prepare_data.py` + `run_eval.sh` path for it (that download has repeatedly stalled on this machine).

#### Optional: `BACKEND=vllm_server` for faster / higher-SM-utilization eval

The default path above (`BACKEND=offline`, implicit) runs VLMEvalKit's serial `LLM.generate()` loop — one request on the GPU at a time, which is why SM utilization looks low during eval. Set `BACKEND=vllm_server` on the same script to instead serve the model with `vllm serve` and drive inference through concurrent HTTP requests (4-5x faster, high SM utilization):

```bash
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit

BACKEND=vllm_server \
MODEL_PATH=<merged checkpoint dir — LoRA auto-merge only works in BACKEND=offline> \
MODEL_NAME=<tag>_server \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K \
GPU_IDS=<free gpu> \
bash shell_scripts/eval_model_temp0_4096.sh
```

Validated on Qwen3-VL-2B-Instruct against the offline path across all 7 benchmarks above (full datasets, real GPT judge scoring): sample-weighted accuracy differs by only **+0.4pp**, every individual benchmark within **±1.5pp** — safe to treat as interchangeable with `BACKEND=offline` at the aggregate-score level. Use `BACKEND=offline` when a result needs to be bit-comparable with historical numbers already in this doc; use `BACKEND=vllm_server` for fast checkpoint screening / sweeps.

Two things that bit us getting this working, both now fixed in the script itself (`VLMEvalKit/shell_scripts/eval_via_vllm_server.sh`), documented here in case a future edit reintroduces them:
- **Every sampling param must be passed explicitly** in the request payload. Anything omitted silently falls back to the served model's own `generation_config.json`, which has no `presence_penalty` field (defaults to 0) — a missing `presence_penalty=1.5` alone caused a real **-3.2pp** gap on MathVista_MINI (runaway repeated text hitting `max_new_tokens` instead of terminating).
- The script needs the proxy **unset** for its own local `127.0.0.1` vLLM calls but **set** (real internet) for the judge/normal-scoring stage at the end — it now saves/restores the proxy vars across that boundary. If you fork this script, don't `unset` proxy vars once at the top and forget to restore them before the judge step; the judge call fails silently or hangs.

See `VLMEvalKit/AGENTS.md` (§ "Optional: `BACKEND=vllm_server`") for the full writeup, including the per-sample agreement investigation (batching and image min_pixels bounds were both tested and ruled out as causes of the residual ~85-90% per-sample text-match rate between the two backends — attributed to differing image-resize implementations, doesn't move the aggregate score).

### ZoomBench — canonical eval (default)

**This is the default ZoomBench pipeline as of 2026-07-08.** It's the native VisionOPD eval (`eval/infer.py` → `eval/judge_qwenlm.py` → `eval/cal_acc.py`) running through `eval/judge_qwenlm.py`'s **v3-fixed judge** (three bugs fixed — see below). Do not use `run_official_eval.sh` (documented further down as a historical/alternate pipeline) unless explicitly asked; it's no longer the default because it hasn't been audited for the same class of judge bugs found here, and its thinking-mode setup is harder to reproduce cheaply.

**Why this is the default now**: `eval/judge_qwenlm.py` had THREE bugs that made ZoomBench (and other MCQ/numeric datasets scored through it) unreliable — same predictions could score anywhere from ~19% to ~53% depending on judge model/version. All three are now fixed in `eval/judge_qwenlm.py` itself (live file, not opt-in — this is not a flag you need to pass). A reproducible snapshot of the fixed script is saved at `eval/judge_qwenlm_zoombench_fixed_20260707.py` — never revert `MCQ_BENCHMARKS` / `extract_first_option()` / `extract_final_number()` / `numeric_match()` in `judge_qwenlm.py` to anything older than this snapshot. Full investigation: `docs/compare_vaopd_0701.md` Phase 4.

1. **`MCQ_BENCHMARKS` was missing `"zoombench"`** → all 620 MCQ questions bypassed the reliable rule-based `first_letter_match` fast path and got routed to an LLM judge with a real false-negative rate (esp. `gpt-5.4-mini`, ~30% FN rate on markdown-formatted answers like `"✅ Correct answer: **C. ..."`). Fixed: `zoombench` added to `MCQ_BENCHMARKS` (also covers `vstar`/`hrbench-4k`/`hrbench-8k`, which were already in the list and benefit from the same fast path).
2. **`extract_first_option()` grabbed the FIRST capital letter in the text**, not the model's actual final answer. Model CoT that enumerates all options ("A. ... B. ... C. ... D. [true answer]") or replies `"Answer: X**"` (the word "Answer" itself starts with capital A) got mis-extracted — silently producing ~10% false positives whenever gt happened to be "A" (confirmed 22/225 in one audit). Fixed: now prefers the last "correct answer is X" / "answer: X" marker, falling back to the *last* enumerated option letter, not the first.
3. **Open-ended "Arabic numeral" answers** (e.g. counting questions) routed the model's full raw CoT into `mathruler.grade_answer()` when there was no `<answer>`/`"Answer:"` marker to truncate on, which almost always failed to match a short numeric GT and fell through to an unreliable LLM judge — disproportionately hurting verbose models (confirmed 37/845 false negatives on one model whose answers were ~20x longer than a terser comparison model, 0 on the terser ones). Fixed: new `extract_final_number()` / `numeric_match()` rule-based fast path, same "prefer the last marker, else the last occurrence" strategy as fix #2.

**Full pipeline (copy-paste template)**:
```bash
# 1. Serve the checkpoint (2B: single GPU; 4B: add --data-parallel-size 2 --gpu-memory-utilization 0.82)
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
vllm serve <ckpt_or_base_model_path> \
  --served-model-name <served_name> --host 0.0.0.0 --port <port> \
  --trust-remote-code --enforce-eager --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --dtype bfloat16 --limit-mm-per-prompt '{"image":8}'

# 2. Infer (temp=0 is hardcoded in infer.py; non-thinking; max_tokens=8192, NOT 32768 —
#    32768 == --max-model-len and will overflow once you add input tokens)
cd Vision-OPD/eval
python3 infer.py --benchmark zoombench --benchmark_json "$PWD/zoombench.json" \
  --out_dir model_answer --model_name <tag> --seed 42 \
  --api_base "http://127.0.0.1:<port>/v1/" --api_key EMPTY \
  --model_id <served_name> --max_tokens 8192 --max_retries 3 --parallel_workers 256

# 3. Judge (v3-fixed judge_qwenlm.py; these exact args)
python3 judge_qwenlm.py --benchmark zoombench --model <tag> \
  --api_base "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl" \
  --api_key "<key from config/key.conf>" --api_type azure --api_version "2024-02-01" \
  --judge_model "gpt-5.4-mini-2026-03-17" --judge_max_tokens 2048

# 4. Accuracy
python3 cal_acc.py --benchmark zoombench --judge_json "judge/zoombench/<tag>_answer.jsonl" \
  --benchmark_json "$PWD/zoombench.json"
```

**Reference numbers** (native eval, temp=0, non-thinking, 845/845, v3-fixed judge) vs the old `run_official_eval.sh` pipeline (thinking, temp=0.7) — kept only as a historical cross-check, not something to reproduce by default:

| Model | Native (v3-fixed, default) | `run_official_eval.sh` (historical) | Gap |
|---|---:|---:|---:|
| Qwen3-VL-4B-base | 44.14% | 41.07% | +3.07pp |
| Qwen3-VL-2B-base | 42.49% | 37.40% | +5.09pp |
| VisionOPD-4B (full) | 52.90% | 48.76% | +4.14pp |
| VisionOPD-2B (full) | 37.51% | 40.00% | −2.49pp |
| noimg-4B | 39.76% | 36.57% | +3.19pp |
| noimg-2B | 36.45% | 37.16% | −0.71pp |
| black-2B | 36.33% | 36.33% | 0pp |
| degrade-2B | 36.33% | 36.80% | −0.47pp |
| qvis-2B | 38.11% | 36.69% | +1.42pp |

All models fall within ±5.1pp of the historical pipeline (down from 15-20pp before the judge fixes); the residual gap is most likely the thinking-vs-non-thinking inference setting, not judge noise.

### `run_official_eval.sh` — historical/alternate pipeline, not the default

**Script**: `Zooming-without-Zooming/mm-eval/run_official_eval.sh`

Kept for cross-checking against the paper's reported number (Qwen3-VL-4B-Instruct = 41.07%, matches paper ~41%), not as the default eval. It has not been audited for judge bugs the way `judge_qwenlm.py` has (see above) — treat its numbers as a secondary reference, not ground truth.

```bash
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Zooming-without-Zooming/mm-eval

bash run_official_eval.sh <model_path> <model_tag> [gpu=0]

# Example — base 2B:
bash run_official_eval.sh \
  /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
  Qwen3-VL-2B-Instruct

# Example — checkpoint:
bash run_official_eval.sh \
  checkpoints/Vision-OPD-noimg-Qwen3-VL-2B-Instruct/global_step_62 \
  VisionOPD-noimg-2B-step62
```

**Pipeline**: thinking mode (temp=0.7, max_tokens=8192) → MathRuler + letter_match → Azure GPT judge (`gpt-5.4-mini`).
Results saved to `Zooming-without-Zooming/mm-eval/judge/zoom-bench/<model_tag>_seed42_answer.json`.

`MAX_MODEL_LEN` defaults to 24576 (covers all 845 samples; one image hits 16432 tokens).

**Important caveat**: this checkpoint's own baked-in `chat_template.jinja` (from RA-VAD/noimg training) unconditionally emits an empty, pre-closed `<think>\n\n</think>\n\n` block regardless of the `enable_thinking` flag passed at inference time — the `_enable_thinking` template variable is set once and never referenced again. So even "thinking mode" here is not real chain-of-thought for this checkpoint; don't assume the paper's higher numbers come from genuine thinking unless you've verified the paper used a differently-templated checkpoint.

### VisionOPD native eval (other benchmarks: vstar, hrbench-4k, hrbench-8k, etc.)

Same three-step pipeline as ZoomBench above (`infer.py` → `judge_qwenlm.py` → `cal_acc.py`), same v3-fixed judge (these datasets were already in `MCQ_BENCHMARKS`, so they were already getting the reliable rule-based fast path once fix #1 landed). Useful for cross-checking against VLMEvalKit's numbers on the same checkpoint, since VLMEvalKit uses a completely different inference/scoring stack (see `docs/compare_vaopd_0701.md` for native-vs-VLMEvalKit comparisons once run).

```
infer.py         reads benchmark JSON, calls /v1/chat/completions, writes JSONL to model_answer/
judge_qwenlm.py  rule-based first, then LLM judge for remainder, writes to judge/<bench>/
cal_acc.py       reads judge JSONL, prints accuracy
```

**Full multi-benchmark eval** (requires a running vLLM server):
```bash
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="<served-model-name>" \
BENCHMARK="vstar,hrbench-4k,hrbench-8k" \
bash eval/run_eval.sh
```

Always pass Azure-mode judge args explicitly (`--azure_api_version`/`api_type=azure`) if invoking `judge_qwenlm.py` directly instead of through `run_eval.sh` — without them, LLM-judge calls to a reasoning judge model (`gpt-5.x`) can fail silently and default every case to `"No"`.

### Benchmark JSON files

| Benchmark | JSON file |
|-----------|-----------|
| zoombench | `zoombench_mcq.json` (native eval) / `zoombench.json` (official pipeline) |
| vstar | `vstar.json` |
| hrbench-4k/8k | `hr_bench_4k.json` / `hr_bench_8k.json` |
| mme-realworld | `MME_RealWorld.json` |

## Checkpoint Paths

All trained checkpoints under:
```
checkpoints/Vision-OPD-<variant>-Qwen3-VL-2B-Instruct/global_step_<N>/
```

Latest step is in `latest_checkpointed_iteration.txt` in the same directory.

ZoomBench results — three judge conditions:

⚠️ The "Azure GPT (native eval)" column below predates the 2026-07-07 `judge_qwenlm.py` fix (see "ZoomBench via native eval — judge bugs fixed" above) and is **unreliable/superseded** — same predictions rejudged with the fixed script score up to 15-20pp higher. Rows already reprocessed with the fixed judge are marked with the corrected number; unreprocessed rows keep the old (untrustworthy) number until rerun.

| Model | Self-judge | Azure GPT (native eval, ⚠️pre-fix unless noted) | **Official pipeline** |
|-------|-----------|-------------------------|-----------------------|
| Qwen3-VL-2B base | 43.79% | ~~31.48%~~ → **42.84%**(fixed judge,2026-07-07) | **37.40%** |
| Qwen3-VL-4B base | 44.14% | ~~34.32%~~ → **44.14%**(fixed judge,2026-07-07) | **41.07%** |
| VisionOPD-baseline (step 65) | 34.67% | 21.89% | — |
| VisionOPD-noimg-2B (step 62) | 49.82% | ~~16.80%~~ → **31.72%**(fixed judge,2026-07-07) | **37.16%** |
| VisionOPD-noimg-seed123-2B (step 62) | — | — | **37.04%** |
| VisionOPD-full-2B (step 65) | 56.92% | ~~19.53%~~ → **35.50%**(fixed judge,2026-07-07) | **40.00%** |
| VisionOPD-black-2B (step 65) | — | 15.86% | **36.33%** |
| VisionOPD-degrade-2B (step 62) | — | 16.57% | **36.80%** |
| VisionOPD-qvis-2B (step 62) | — | 16.92% | **36.69%** |
| VisionOPD-full-4B (step 65) | — | **52.90%**(fixed judge,2026-07-07) | 48.76% |
| VisionOPD-noimg-4B (step 62) | — | **35.38%**(fixed judge,2026-07-07) | 36.57% |
| 4Bteacher→2Bstudent-noimg (step 62) | — | 22.13% | — |

**Official pipeline** = `run_official_eval.sh` (thinking, temp=0.7, Azure GPT + letter_match).
Self-judge is ~3–4 pp above official; VisionOPD-trained models show extreme gaps (30–37 pp)
suggesting reward hacking: RL reward used self-judge, training gamed the judge rather than
improving true visual ability.

## Training

Entry point: `bash scripts/run_vision_opd.sh` or `bash scripts/run_experiment_grpo_baseline.sh`.

Key env vars: `MODEL_PATH`, `EXPERIMENT_NAME`, `CUDA_VISIBLE_DEVICES`, `TRAINER_N_GPUS_PER_NODE`, `TRAIN_BATCH_SIZE`, `ROLLOUT_N`, `LR`, `TRAINER_TOTAL_EPOCHS`, `TRAINER_SAVE_FREQ`.

## Do Not Commit

Checkpoints, rollouts, downloaded datasets, API keys, `.env` files, generated JSONL answers.
