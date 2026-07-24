#!/bin/bash
set -euo pipefail

################################################################################
# Qwen3.5 (Vision-OPD) environment setup — verified 2026-07-12
#
# Replaces the old ad-hoc wheel-install block:
#   pip install --no-deps transformers-5.3.0.dev0-py3-none-any.whl
#   pip install flash_linear_attention-0.4.2-py3-none-any.whl
#   pip install causal_conv1d-1.6.0-cp311-cp311-linux_x86_64.whl
#
# IMPORTANT — run this AFTER trail_setup.sh / opsd/unsup-opsd/setup.sh, not
# before. unsup-opsd/setup.sh runs `pip install -r requirements.txt` which
# pins transformers==4.57.3 (no Qwen3.5 support) — running the old wheel
# install block *before* that step meant it got silently overwritten. This
# script must run last so its versions are the ones left standing.
#
# See Vision-OPD/requirements_qwen35.txt for the full version rationale.
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Torch/vllm/transformers/etc, matched to a working combo (torch 2.10.0 <->
# vllm 0.18.0 CUDA-extension ABI match). --no-deps to avoid pip pulling in
# an unrelated resolver-driven version cascade.
python3 -m pip install --no-deps -r "${SCRIPT_DIR}/../requirements_qwen35.txt"

# flash-attn and causal-conv1d have compiled CUDA extensions pinned to a
# torch build, so they can't be plain pip-installed — must build from source
# against whatever torch was just installed above. --force-reinstall is
# required: without it, pip sees an old install "already satisfies" the
# unversioned/older requirement and silently skips the rebuild, leaving a
# stale binary that fails at runtime with `undefined symbol` (not at import
# time — only when the rollout engine actually initializes).
python3 -m pip install --user --no-build-isolation --force-reinstall --no-deps flash-attn
python3 -m pip install --user --no-build-isolation --force-reinstall --no-deps causal-conv1d==1.6.1
# Verified 2026-07-13: the unversioned flash-attn install above resolved to
# flash-attn==2.8.3.post1 in this environment.

echo "Qwen3.5 env setup done. Verify with:"
echo "  python3 -c \"import torch, vllm, flash_attn, causal_conv1d; from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration; print('OK')\""
