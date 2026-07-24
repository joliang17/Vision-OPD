#!/usr/bin/env python3
"""Sample N rollouts per question from a vLLM OpenAI-compatible server and
flag questions where the rollouts disagree (some correct, some wrong).

Reads a verl-format parquet (prompt/images/reward_model columns), sends each
question with n rollouts to the server, scores each rollout with the
project's MCQ exact-match reward, and writes one JSONL record per question
with per-rollout correctness plus a `mixed` flag.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcq_exact_reward import compute_score  # noqa: E402


def image_bytes_to_data_uri(raw: bytes) -> str:
    img = Image.open(BytesIO(raw)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(prompt_msgs, images) -> list[dict]:
    data_uris = [image_bytes_to_data_uri(im["bytes"]) for im in images]
    messages = []
    for m in prompt_msgs:
        role = m["role"]
        content = m["content"]
        if role != "user" or "<image>" not in content:
            messages.append({"role": role, "content": content})
            continue
        parts = content.split("<image>")
        content_list = []
        for i, part in enumerate(parts):
            if part:
                content_list.append({"type": "text", "text": part})
            if i < len(parts) - 1 and i < len(data_uris):
                content_list.append({"type": "image_url", "image_url": {"url": data_uris[i]}})
        messages.append({"role": role, "content": content_list})
    return messages


def load_done_ids(out_path: str) -> set[str]:
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--dataset-tag", default="")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df.iloc[args.shard_idx :: args.num_shards].reset_index(drop=True)

    done_ids = load_done_ids(args.out)
    lock = threading.Lock()
    out_f = open(args.out, "a")

    client = OpenAI(api_key="EMPTY", base_url=args.api_base)

    def process_row(idx_row):
        idx, row = idx_row
        rec_id = f"{args.dataset_tag}:{args.shard_idx}:{idx}"
        if rec_id in done_ids:
            return None
        gt = row["reward_model"]["ground_truth"] if row["reward_model"] is not None else None
        try:
            messages = build_messages(list(row["prompt"]), list(row["images"]) if row["images"] is not None else [])
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                n=args.n_rollouts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            rollouts = []
            correct_count = 0
            for choice in resp.choices:
                text = choice.message.content or ""
                score = compute_score(solution_str=text, ground_truth=gt)
                is_correct = bool(score["acc"])
                correct_count += int(is_correct)
                rollouts.append({"text": text, "pred": score["pred"], "correct": is_correct})
            record = {
                "id": rec_id,
                "dataset": args.dataset_tag,
                "data_source": row.get("data_source"),
                "ability": row.get("ability"),
                "ground_truth": gt,
                "n_rollouts": len(rollouts),
                "correct_count": correct_count,
                "mixed": 0 < correct_count < len(rollouts),
                "rollouts": rollouts,
            }
        except Exception as e:
            record = {"id": rec_id, "dataset": args.dataset_tag, "error": str(e)}
        return record

    tasks = list(df.iterrows())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_row, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{args.dataset_tag} shard{args.shard_idx}"):
            record = fut.result()
            if record is None:
                continue
            with lock:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

    out_f.close()


if __name__ == "__main__":
    main()
