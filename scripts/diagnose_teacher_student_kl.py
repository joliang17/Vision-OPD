#!/usr/bin/env python3
"""Diagnostic: is the per-token teacher-student KL on HIGH-RA-weight tokens any larger than on
low-weight tokens?

Motivation (docs/compare_vaopd_0701.md Phase 2-核心 第九轮 follow-up): three weighting sources
(uniform / logprob-gap / attention) are indistinguishable downstream. One candidate explanation:
the EMA teacher sees the SAME input as the student and its weights are a slow-moving average of
the student's, so the per-token distillation target barely differs from the student anywhere —
including on perfectly-identified visual tokens. If true, token weighting multiplies a
near-zero signal and the choice of weights cannot matter.

This script loads the saved EMA teacher + matching student checkpoint (both merged HF format),
replays real rollout responses through both on the SAME hi (image-present) input, computes the
full-vocab per-token KL(teacher || student) at the training temperature, computes the standard
logprob-gap RA weights from the teacher's hi/ctrl pair, and reports KL bucketed by weight.

Interpretation:
  - KL(high-weight bucket) ~= KL(low/zero-weight bucket) ~= tiny  -> "token found, but no signal
    there" — the identical-input EMA teacher provides nothing token-specific to transfer.
  - KL(high-weight bucket) >> others -> signal exists but is diluted; weighting scheme/loss scale
    is the problem instead.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ra_tokens import (  # noqa: E402
    build_mode_messages,
    build_row_index,
    load_rollouts,
    normalize_text,
    rollout_prompt_text,
)
from verl.trainer.ppo.ra_vad import compute_ra_weights  # noqa: E402


def robust_prompt_text(input_text: str) -> str:
    """Like rollout_prompt_text, but also handles dumps whose chat text includes a leading
    system turn (e.g. the virl39k think/answer-format system prompt) before 'user\n\n'."""
    text = input_text
    if "user\n" in text:
        text = text.split("user\n", 1)[1]
    marker = "\nassistant\n"
    if marker in text:
        text = text.split(marker, 1)[0]
    return normalize_text(text)


@torch.no_grad()
def response_full_logprobs(model, processor, messages, response, device, temperature=1.0):
    """Full-vocab log-probs at every response-token position, plus chosen-token logp."""
    images = []
    for m in messages:
        for item in m.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                images.append(item["image"])
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_text = prompt_text + response

    prompt_inputs = processor(text=[prompt_text], images=images or None, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=images or None, return_tensors="pt")
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    full_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_inputs.items()}
    input_ids = full_inputs["input_ids"][0]
    if prompt_len >= input_ids.numel():
        return None, None, None

    out = model(**full_inputs, use_cache=False)
    logits = out.logits[0, prompt_len - 1 : input_ids.numel() - 1, :].float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)  # (n_resp, vocab)
    target_ids = input_ids[prompt_len:]
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return log_probs, token_logps, target_ids.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=2.0, help="Match training ra_temperature.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.student_path, trust_remote_code=True)
    student = AutoModelForImageTextToText.from_pretrained(
        args.student_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    student.eval()
    teacher = AutoModelForImageTextToText.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    teacher.eval()

    df = pd.read_parquet(args.parquet)
    row_index = build_row_index(df)
    records = load_rollouts(Path(args.rollout_dir), args.max_samples)

    rows = []
    misses = 0
    for sample_id, record in enumerate(tqdm(records, desc="diagnose")):
        key = (robust_prompt_text(record["input"]), str(record.get("gts", "")))
        row_idx = row_index.get(key)
        if row_idx is None:
            misses += 1
            continue
        row = df.iloc[row_idx]
        response = str(record.get("output", ""))
        if not response.strip():
            continue
        try:
            hi_messages, ctrl_messages = build_mode_messages(row, "noimg")
        except Exception:
            continue
        try:
            t_logprobs, t_token_logps, ids = response_full_logprobs(
                teacher, processor, hi_messages, response, device, args.temperature
            )
            if t_logprobs is None:
                continue
            _, t_ctrl_token_logps, ctrl_ids = response_full_logprobs(
                teacher, processor, ctrl_messages, response, device, args.temperature
            )
            s_logprobs, _, _ = response_full_logprobs(
                student, processor, hi_messages, response, device, args.temperature
            )
        except Exception as e:
            print(f"sample {sample_id} failed: {e}")
            continue
        n = min(t_logprobs.shape[0], t_ctrl_token_logps.shape[0], s_logprobs.shape[0])
        if ids[:n] != ctrl_ids[:n]:
            shared = 0
            for a, b in zip(ids, ctrl_ids):
                if a != b:
                    break
                shared += 1
            n = shared
        if n == 0:
            continue

        # KL(teacher || student), full vocab, per token
        kl = F.kl_div(s_logprobs[:n], t_logprobs[:n], reduction="none", log_target=True).sum(-1)

        ra_raw = (t_token_logps[:n] - t_ctrl_token_logps[:n]).unsqueeze(0)
        mask = torch.ones_like(ra_raw)
        weights, _ = compute_ra_weights(
            logp_hi=t_token_logps[:n].unsqueeze(0),
            logp_ctrl=t_ctrl_token_logps[:n].unsqueeze(0),
            response_mask=mask,
        )
        weights = weights[0]
        for t in range(n):
            rows.append(
                {
                    "sample_id": sample_id,
                    "t": t,
                    "kl": float(kl[t].item()),
                    "ra_weight": float(weights[t].item()),
                    "ra_raw": float(ra_raw[0, t].item()),
                }
            )

    print(f"matched={len(records) - misses} misses={misses} tokens={len(rows)}")
    out = pd.DataFrame(rows)
    if args.output_csv:
        out.to_csv(args.output_csv, index=False)
        print(f"wrote {args.output_csv}")

    zero = out[out["ra_weight"] == 0]
    pos = out[out["ra_weight"] > 0]
    print(f"\n=== KL(teacher || student) at T={args.temperature}, bucketed by ra_weight ===")
    print(f"all tokens        : n={len(out):6d}  KL mean={out['kl'].mean():.6f}  median={out['kl'].median():.6f}")
    print(f"weight == 0       : n={len(zero):6d}  KL mean={zero['kl'].mean():.6f}  median={zero['kl'].median():.6f}")
    if len(pos):
        qs = pos["ra_weight"].quantile([0.25, 0.5, 0.75, 0.9]).values
        buckets = [
            ("weight (0, q25]", pos[pos["ra_weight"] <= qs[0]]),
            ("weight (q25,q50]", pos[(pos["ra_weight"] > qs[0]) & (pos["ra_weight"] <= qs[1])]),
            ("weight (q50,q75]", pos[(pos["ra_weight"] > qs[1]) & (pos["ra_weight"] <= qs[2])]),
            ("weight (q75,q90]", pos[(pos["ra_weight"] > qs[2]) & (pos["ra_weight"] <= qs[3])]),
            ("weight  > q90    ", pos[pos["ra_weight"] > qs[3]]),
        ]
        for label, b in buckets:
            print(f"{label}: n={len(b):6d}  KL mean={b['kl'].mean():.6f}  median={b['kl'].median():.6f}")
    print(f"\ncorr(ra_weight, kl) = {out['ra_weight'].corr(out['kl']):.4f}")
    print(f"corr(ra_raw, kl)    = {out['ra_raw'].corr(out['kl']):.4f}")


if __name__ == "__main__":
    main()
