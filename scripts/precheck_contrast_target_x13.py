#!/usr/bin/env python3
"""X13: guarded-tilting 零训练 precheck 扩展（2026-07-17，queue.md X13）。

在 precheck_contrast_target.py 基础上补三个空白：
  1. β=0（完全去掉 plausibility mask）——旧扫描只有 β∈{0.1,0.05}
  2. EOS tilt 豁免开关（训练里 exclude_token_ids=[151643,151645] 保留原 lp_hi 打分不参与倾斜；
     旧 precheck 从未实现豁免，等价于"无豁免"形态——本脚本把两种形态都算）
  3. 新统计：垃圾 argmax 率（新 argmax 在 p_hi 下 rank>10 的占比）+ EOS 概率抬升（p_target−p_hi 在
     EOS token 集上的均值差）

配置网格：ctrl=black，α=1.0（contrast-标准），(β, 豁免) ∈ {0.1,0} × {on,off}，
其中 (0.1, on) = 训练默认，作为参照行。
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
from analyze_ra_tokens import build_mode_messages, build_row_index, load_rollouts  # noqa: E402
from precheck_contrast_target import response_logits, robust_prompt_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--betas", default="0.1,0")
    parser.add_argument("--ctrl-mode", default="black")
    parser.add_argument("--exclude-token-ids", default="151643,151645")
    parser.add_argument("--garbage-rank", type=int, default=10)
    parser.add_argument("--output-dir", default="analysis_outputs/contrast_target_precheck_x13")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.teacher_path, trust_remote_code=True)
    teacher = AutoModelForImageTextToText.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    teacher.eval()

    a = args.alpha
    betas = [float(x) for x in args.betas.split(",")]
    excl_ids = [int(x) for x in args.exclude_token_ids.split(",") if x.strip()]
    mode = args.ctrl_mode
    configs = [(b, ex) for b in betas for ex in ("on", "off")]

    df = pd.read_parquet(args.parquet)
    row_index = build_row_index(df)
    records = load_rollouts(Path(args.rollout_dir), args.max_samples)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {c: {"n": 0, "changed": 0, "kl_sum": 0.0, "garbage": 0, "eos_dp_sum": 0.0,
                 "eos_argmax_gained": 0} for c in configs}
    change_pairs = {c: Counter() for c in configs}

    misses = 0
    for sample_id, record in enumerate(tqdm(records, desc="x13")):
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
            hi_messages, ctrl_messages = build_mode_messages(row, mode)
            logits_hi, ids = response_logits(teacher, processor, hi_messages, response, device)
            if logits_hi is None:
                continue
            logits_ctrl, ctrl_ids = response_logits(teacher, processor, ctrl_messages, response, device)
            if logits_ctrl is None:
                continue
        except Exception as e:  # noqa: BLE001
            print(f"sample {sample_id} failed: {e}")
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

        lh = logits_hi[:n]
        lp_hi = torch.log_softmax(lh, dim=-1)
        p_hi = lp_hi.exp()
        argmax_hi = lp_hi.argmax(-1)
        tilted_base = lh + a * (lh - logits_ctrl[:n])
        # p_hi 下的降序 rank（0=最大）
        for b, ex in configs:
            tilted = tilted_base
            if ex == "on" and excl_ids:
                tilted = tilted_base.clone()
                tilted[:, excl_ids] = lh[:, excl_ids]
            if b > 0:
                max_lp = lp_hi.max(dim=-1, keepdim=True).values
                mask = lp_hi >= (max_lp + torch.log(torch.tensor(b, device=device)))
                tilted = tilted.masked_fill(~mask, float("-inf"))
            lp_t = torch.log_softmax(tilted, dim=-1)
            p_t = lp_t.exp()
            argmax_t = lp_t.argmax(-1)
            changed = argmax_t != argmax_hi
            kl = F.kl_div(lp_hi, p_t, reduction="none").sum(-1)
            # 垃圾率：新 argmax 在 p_hi 下 rank > garbage_rank
            new_lp = lp_hi.gather(-1, argmax_t.unsqueeze(-1))
            rank = (lp_hi > new_lp).sum(-1)
            garbage = changed & (rank > args.garbage_rank)
            # EOS 概率抬升
            eos_dp = (p_t[:, excl_ids].sum(-1) - p_hi[:, excl_ids].sum(-1)) if excl_ids else torch.zeros(n, device=device)
            eos_gain = changed & torch.isin(argmax_t, torch.tensor(excl_ids, device=device)) if excl_ids else torch.zeros(n, dtype=torch.bool, device=device)

            st = stats[(b, ex)]
            st["n"] += n
            st["changed"] += int(changed.sum())
            st["kl_sum"] += float(kl.sum())
            st["garbage"] += int(garbage.sum())
            st["eos_dp_sum"] += float(eos_dp.sum())
            st["eos_argmax_gained"] += int(eos_gain.sum())
            for t in torch.nonzero(changed).squeeze(-1).detach().cpu().tolist():
                old = processor.tokenizer.decode([int(argmax_hi[t])]).strip()
                new = processor.tokenizer.decode([int(argmax_t[t])]).strip()
                change_pairs[(b, ex)][(old, new)] += 1

    print(f"matched={len(records) - misses} misses={misses}")
    rows = []
    for (b, ex), st in stats.items():
        if st["n"] == 0:
            continue
        rows.append({
            "ctrl": mode, "alpha": a, "beta": b, "eos_exempt": ex,
            "positions": st["n"],
            "argmax_change_rate": st["changed"] / st["n"],
            "mean_KL(target||p_hi)": st["kl_sum"] / st["n"],
            "garbage_argmax_rate(rank>%d)" % args.garbage_rank: st["garbage"] / st["n"],
            "garbage_share_of_changes": st["garbage"] / max(1, st["changed"]),
            "mean_eos_prob_delta": st["eos_dp_sum"] / st["n"],
            "eos_argmax_gained": st["eos_argmax_gained"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    for c, counter in change_pairs.items():
        b, ex = c
        pd.DataFrame([{"old": o, "new": nw, "count": cnt} for (o, nw), cnt in counter.most_common()]).to_csv(
            out_dir / f"changes_beta{b}_exempt{ex}.csv", index=False)
        print(f"\n=== top changes beta={b} exempt={ex} ===")
        for (o, nw), cnt in counter.most_common(15):
            print(f"  {cnt:4d}  {o!r} -> {nw!r}")


if __name__ == "__main__":
    main()
