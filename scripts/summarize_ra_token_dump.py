#!/usr/bin/env python3
"""Decode + summarize the per-step dumps written by
DataParallelPPOActor._dump_ra_token_weights (opt-in via self_distillation.ra_token_dump_dir).

Each dump is a torch.save'd dict of {response_ids, ra_weights, response_mask, global_step, rank,
...} for a few samples. This script walks all dumps under a dump dir, decodes response_ids with
the checkpoint's tokenizer, and prints/saves the top-weighted tokens per step so you can watch
"what does the model currently think is a high-weight token" evolve over training without
re-running an offline analysis pass.
"""

import argparse
import glob
import os
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True, help="ra_token_dump_dir/<experiment_name> from training.")
    parser.add_argument("--tokenizer-path", required=True, help="Path to the checkpoint/base model tokenizer.")
    parser.add_argument("--top-k", type=int, default=15, help="Top-K tokens to print per step.")
    parser.add_argument("--output-csv", default=None, help="Optional path to save the full per-token table.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    files = sorted(
        glob.glob(os.path.join(args.dump_dir, "*.pt")),
        key=lambda p: (int(os.path.basename(p).split(".")[0]), os.path.basename(p)),
    )
    if not files:
        raise SystemExit(f"No .pt dumps found under {args.dump_dir}")

    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        step = d["global_step"]
        response_ids = d["response_ids"]  # (n, resp_len)
        ra_weights = d["ra_weights"]  # (n, resp_len)
        response_mask = d["response_mask"]  # (n, resp_len)
        for i in range(response_ids.shape[0]):
            for t in range(response_ids.shape[1]):
                if response_mask[i, t].item() <= 0:
                    continue
                token_id = int(response_ids[i, t].item())
                text = tokenizer.decode([token_id], skip_special_tokens=False)
                rows.append(
                    {
                        "step": step,
                        "sample": i,
                        "t": t,
                        "token_id": token_id,
                        "text": text,
                        "ra_weight": float(ra_weights[i, t].item()),
                    }
                )

    df = pd.DataFrame(rows)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"wrote {args.output_csv} ({len(df)} rows)")

    steps = sorted(df["step"].unique())
    for step in steps:
        step_df = df[df["step"] == step]
        top = step_df.sort_values("ra_weight", ascending=False).head(args.top_k)
        print(f"\n=== step {step} — top {args.top_k} tokens by ra_weight ===")
        print(top[["text", "ra_weight", "sample", "t"]].to_string(index=False))

    # Word-level trend across steps, for words seen at multiple steps.
    df["text_norm"] = df["text"].astype(str).str.strip().str.lower()
    trend = defaultdict(dict)
    for step in steps:
        step_df = df[df["step"] == step]
        means = step_df.groupby("text_norm")["ra_weight"].mean()
        for word, mean in means.items():
            trend[word][step] = mean
    trend_rows = [{"text": w, **{f"step_{s}": trend[w].get(s) for s in steps}} for w in trend]
    trend_df = pd.DataFrame(trend_rows)
    print("\n=== per-word mean ra_weight across steps (words appearing at multiple steps) ===")
    multi_step = trend_df[trend_df[[f"step_{s}" for s in steps]].notna().sum(axis=1) > 1]
    print(multi_step.to_string(index=False))


if __name__ == "__main__":
    main()
