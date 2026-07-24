#!/usr/bin/env python3
"""V1 (arXiv plan): human-readable visualization of what the contrast-sharpened
target distribution actually decodes to, position by position.

For each sampled rollout, teacher-force the EMA teacher under hi (real image)
and ctrl (black image) conditions, build the training target
    target = softmax(logits_hi + alpha * (logits_hi - logits_ctrl))
              restricted to {v : p_hi(v) >= beta * max p_hi}
and emit a markdown report showing, per response position:
    actual sampled token | argmax p_hi | argmax target | top-k target tokens
    | ra_raw (chosen-token logprob gap) | tilt-changed flag
plus per-sample summaries. This is the "decode the target into human language"
artifact for the paper's qualitative/visualization ablation.

Matches training semantics from verl/trainer/ppo/ra_vad.py build_contrast_target:
alpha=1.0, beta=0.1, black ctrl, tilt exemption for <|im_end|>/<|endoftext|>.
(Temperature: report both T=1 raw and T=2 softened argmax rarely differ; we use
raw logits like scripts/precheck_contrast_target.py for readability.)
"""

import argparse
import html
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ra_tokens import build_mode_messages, build_row_index, load_rollouts  # noqa: E402
from precheck_contrast_target import response_logits, robust_prompt_text  # noqa: E402

EXCLUDE_TOKEN_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|> — tilt-exempt in training


def build_target(logits_hi, logits_ctrl, alpha, beta, device):
    lp_hi = torch.log_softmax(logits_hi, dim=-1)
    tilted = logits_hi + alpha * (logits_hi - logits_ctrl)
    # tilt exemption: keep original hi logits for termination tokens
    for tid in EXCLUDE_TOKEN_IDS:
        tilted[:, tid] = logits_hi[:, tid]
    max_lp = lp_hi.max(dim=-1, keepdim=True).values
    mask = lp_hi >= (max_lp + torch.log(torch.tensor(beta, device=device)))
    masked = tilted.masked_fill(~mask, float("-inf"))
    lp_target = torch.log_softmax(masked, dim=-1)
    return lp_hi, lp_target


def tok(processor, tid):
    s = processor.tokenizer.decode([int(tid)])
    return html.escape(repr(s)[1:-1]) if s else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", required=True)
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-samples", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ctrl-mode", default="black")
    ap.add_argument("--output", default="analysis_outputs/target_decoding_vis/report.md")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.teacher_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    df = pd.read_parquet(args.parquet)
    row_index = build_row_index(df)
    records = load_rollouts(Path(args.rollout_dir), args.max_samples * 3)  # extra for misses

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Target-distribution decoding visualization\n",
        f"teacher={args.teacher_path}\nrollouts={args.rollout_dir}\n"
        f"alpha={args.alpha} beta={args.beta} ctrl={args.ctrl_mode} topk={args.topk}\n",
        "Legend: pos | actual sampled token | argmax p_hi | argmax target "
        "(**bold** = tilt changed the argmax) | top-k target tokens (prob) | ra_raw\n",
    ]

    done = 0
    for record in tqdm(records, desc="vis"):
        if done >= args.max_samples:
            break
        key = (robust_prompt_text(record["input"]), str(record.get("gts", "")))
        row_idx = row_index.get(key)
        if row_idx is None:
            continue
        row = df.iloc[row_idx]
        response = str(record.get("output", ""))
        if not response.strip():
            continue
        try:
            hi_msgs, ctrl_msgs = build_mode_messages(row, args.ctrl_mode)
            logits_hi, ids = response_logits(model, processor, hi_msgs, response, device)
            if logits_hi is None:
                continue
            logits_ctrl, ctrl_ids = response_logits(model, processor, ctrl_msgs, response, device)
            if logits_ctrl is None:
                continue
        except Exception as e:  # noqa: BLE001
            print(f"skip: {e}")
            continue
        n = min(logits_hi.shape[0], logits_ctrl.shape[0], len(ids), len(ctrl_ids))
        if ids[:n] != ctrl_ids[:n]:
            shared = 0
            for x, y in zip(ids, ctrl_ids):
                if x != y:
                    break
                shared += 1
            n = shared
        if n == 0:
            continue

        lp_hi, lp_target = build_target(logits_hi[:n].float(), logits_ctrl[:n].float(),
                                        args.alpha, args.beta, device)
        tgt_ids = torch.tensor(ids[:n], device=device)
        ra_raw = (lp_hi.gather(-1, tgt_ids.unsqueeze(-1))
                  - torch.log_softmax(logits_ctrl[:n].float(), dim=-1).gather(-1, tgt_ids.unsqueeze(-1))
                  ).squeeze(-1)
        am_hi, am_t = lp_hi.argmax(-1), lp_target.argmax(-1)
        changed = (am_t != am_hi)
        probs_t = lp_target.exp()
        topv, topi = probs_t.topk(args.topk, dim=-1)

        q = str(row.get("problem", row.get("question", "")))[:500]
        lines.append(f"\n---\n\n## Sample {done} (rollout row {row_idx})\n")
        lines.append(f"**Question:** {q}\n\n**GT:** {record.get('gts','')}\n")
        lines.append(f"**Sampled response ({n} tok):** {response[:600]}\n")
        lines.append(f"**Positions where tilt changes argmax: {int(changed.sum())}/{n} "
                     f"({100*float(changed.float().mean()):.1f}%)**\n")
        lines.append("\n| pos | actual | argmax p_hi | argmax target | top-k target | ra_raw |")
        lines.append("|---|---|---|---|---|---|")
        for t in range(n):
            mark = "**" if bool(changed[t]) else ""
            top_str = ", ".join(
                f"`{tok(processor, topi[t, j])}`({topv[t, j]:.2f})" for j in range(args.topk)
            )
            lines.append(
                f"| {t} | `{tok(processor, ids[t])}` | `{tok(processor, am_hi[t])}` "
                f"| {mark}`{tok(processor, am_t[t])}`{mark} | {top_str} | {ra_raw[t]:+.2f} |"
            )
        done += 1

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} with {done} samples")


if __name__ == "__main__":
    main()
