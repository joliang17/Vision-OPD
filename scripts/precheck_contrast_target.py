#!/usr/bin/env python3
"""Zero-training pre-check for the contrast-sharpened distillation target
(docs/compare_vaopd_0701.md Phase 2-核心 第十轮 planning).

Computes, on real on-policy rollout samples, the proposed target
    target ∝ softmax(logits_hi + α (logits_hi − logits_ctrl))  restricted to the
    plausibility set {w : p_hi(w) ≥ β max p_hi}
and reports — WITHOUT training anything — whether the redistribution looks like signal
(boosting visual-content tokens at genuinely image-dependent positions) or like the failure
modes we're worried about (boosting opening-phrase/style tokens or garbage).

Metrics per (ctrl_mode, α, β):
  - argmax-change rate: fraction of response positions where argmax(target) != argmax(p_hi)
  - mean KL(target || p_hi): how hard the tilt is
  - where changes land: fraction of changed positions in the top ra_raw quartile
  - what changes: top (old-argmax -> new-argmax) decoded token pairs

ctrl_mode covers both "noimg" (what training currently uses; known style-confounded) and
"black" (established cleaner in earlier rounds) so the two can be compared directly.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

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
)


def robust_prompt_text(input_text: str) -> str:
    text = input_text
    if "user\n" in text:
        text = text.split("user\n", 1)[1]
    marker = "\nassistant\n"
    if marker in text:
        text = text.split(marker, 1)[0]
    return normalize_text(text)


@torch.no_grad()
def response_logits(model, processor, messages, response, device):
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
        return None, None
    out = model(**full_inputs, use_cache=False)
    logits = out.logits[0, prompt_len - 1 : input_ids.numel() - 1, :].float()
    return logits, input_ids[prompt_len:].detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", required=True, help="EMA teacher (used for both hi and ctrl branches, matching training).")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--alphas", default="0.5,1.0,2.0")
    parser.add_argument("--betas", default="0.1,0.05")
    parser.add_argument("--ctrl-modes", default="noimg,black")
    parser.add_argument("--output-dir", default="analysis_outputs/contrast_target_precheck")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.teacher_path, trust_remote_code=True)
    teacher = AutoModelForImageTextToText.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    teacher.eval()

    alphas = [float(x) for x in args.alphas.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    ctrl_modes = [m.strip() for m in args.ctrl_modes.split(",")]

    df = pd.read_parquet(args.parquet)
    row_index = build_row_index(df)
    records = load_rollouts(Path(args.rollout_dir), args.max_samples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {(m, a, b): {"n": 0, "changed": 0, "kl_sum": 0.0, "changed_top_raq": 0}
             for m in ctrl_modes for a in alphas for b in betas}
    change_pairs = {(m, a, b): Counter() for m in ctrl_modes for a in alphas for b in betas}
    ra_raw_all = {m: [] for m in ctrl_modes}

    misses = 0
    for sample_id, record in enumerate(tqdm(records, desc="precheck")):
        key = (robust_prompt_text(record["input"]), str(record.get("gts", "")))
        row_idx = row_index.get(key)
        if row_idx is None:
            misses += 1
            continue
        row = df.iloc[row_idx]
        response = str(record.get("output", ""))
        if not response.strip():
            continue

        for mode in ctrl_modes:
            try:
                hi_messages, ctrl_messages = build_mode_messages(row, mode)
                logits_hi, ids = response_logits(teacher, processor, hi_messages, response, device)
                if logits_hi is None:
                    continue
                logits_ctrl, ctrl_ids = response_logits(teacher, processor, ctrl_messages, response, device)
                if logits_ctrl is None:
                    continue
            except Exception as e:
                print(f"sample {sample_id} mode {mode} failed: {e}")
                continue
            n = min(logits_hi.shape[0], logits_ctrl.shape[0])
            if ids[:n] != ctrl_ids[:n]:
                shared = 0
                for x, y in zip(ids, ctrl_ids):
                    if x != y:
                        break
                    shared += 1
                n = shared
            if n == 0:
                continue

            lp_hi = torch.log_softmax(logits_hi[:n], dim=-1)
            lp_ctrl = torch.log_softmax(logits_ctrl[:n], dim=-1)
            target_ids_t = torch.tensor(ids[:n], device=device)
            ra_raw = (lp_hi.gather(-1, target_ids_t.unsqueeze(-1)) - lp_ctrl.gather(-1, target_ids_t.unsqueeze(-1))).squeeze(-1)
            ra_raw_all[mode].extend(ra_raw.detach().cpu().tolist())
            raq75 = torch.quantile(ra_raw, 0.75)

            argmax_hi = lp_hi.argmax(-1)
            contrast = logits_hi[:n] - logits_ctrl[:n]
            for a in alphas:
                tilted = logits_hi[:n] + a * contrast
                for b in betas:
                    # plausibility mask relative to p_hi
                    max_lp = lp_hi.max(dim=-1, keepdim=True).values
                    mask = lp_hi >= (max_lp + torch.log(torch.tensor(b, device=device)))
                    masked = tilted.masked_fill(~mask, float("-inf"))
                    lp_target = torch.log_softmax(masked, dim=-1)
                    argmax_t = lp_target.argmax(-1)
                    changed = argmax_t != argmax_hi
                    kl = F.kl_div(lp_hi, lp_target.exp(), reduction="none").sum(-1)  # KL(target||p_hi)

                    st = stats[(mode, a, b)]
                    st["n"] += n
                    st["changed"] += int(changed.sum().item())
                    st["kl_sum"] += float(kl.sum().item())
                    st["changed_top_raq"] += int((changed & (ra_raw > raq75)).sum().item())

                    for t in torch.nonzero(changed).squeeze(-1).detach().cpu().tolist():
                        old = processor.tokenizer.decode([int(argmax_hi[t])]).strip()
                        new = processor.tokenizer.decode([int(argmax_t[t])]).strip()
                        change_pairs[(mode, a, b)][(old, new)] += 1

    print(f"matched={len(records) - misses} misses={misses}")
    rows = []
    for (mode, a, b), st in stats.items():
        if st["n"] == 0:
            continue
        rows.append({
            "ctrl_mode": mode, "alpha": a, "beta": b,
            "positions": st["n"],
            "argmax_change_rate": st["changed"] / st["n"],
            "mean_KL(target||p_hi)": st["kl_sum"] / st["n"],
            "changed_in_top_ra_raw_quartile": (st["changed_top_raq"] / max(1, st["changed"])),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))

    for (mode, a, b), counter in change_pairs.items():
        if a == 1.0 and counter:
            print(f"\n=== top argmax changes (old -> new), ctrl={mode}, alpha={a}, beta={b} ===")
            for (old, new), cnt in counter.most_common(25):
                print(f"  {cnt:4d}  {old!r} -> {new!r}")
            pd.DataFrame(
                [{"old": o, "new": nw, "count": c} for (o, nw), c in counter.most_common()]
            ).to_csv(out_dir / f"changes_{mode}_a{a}_b{b}.csv", index=False)


if __name__ == "__main__":
    main()
