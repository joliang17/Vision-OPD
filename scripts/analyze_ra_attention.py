#!/usr/bin/env python3
"""Sanity check: does attention-to-image-token concentration highlight the 'right'
(content/visual) tokens, as opposed to the logp_hi - logp_ctrl RA-weight signal which
was found to be dominated by opening-phrase discourse markers (see
docs/compare_vaopd_0701.md, Phase 2-核心 三、四、五轮)?

For each response token (generated with the real image present), extract attention mass
from that token's query position to the image-token key span, averaged over layers/heads.
Aggregate by token text and compare against the same aggregation done for ra_weight.

Reuses rollout matching / message-building helpers from analyze_ra_tokens.py to guarantee
identical sample matching against the same rollout dumps.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ra_tokens import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    build_row_index,
    load_rollouts,
    make_prompt_messages,
    rollout_prompt_text,
)

IMAGE_TOKEN_ID = 151655


@torch.no_grad()
def response_image_attention(model, processor, messages, response, device):
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
        return [], [], np.empty(0)

    image_mask = (input_ids == IMAGE_TOKEN_ID).cpu().numpy()
    if image_mask.sum() == 0:
        return [], [], np.empty(0)

    outputs = model(**full_inputs, output_attentions=True, use_cache=False)
    n_layers = len(outputs.attentions)
    seq_len = input_ids.numel()

    q_start, q_end = prompt_len - 1, seq_len - 1
    if q_end <= q_start:
        return [], [], np.empty(0)
    attn_sum = torch.zeros(q_end - q_start, dtype=torch.float32, device=device)
    for layer_attn in outputs.attentions:
        head_mean = layer_attn[0].mean(dim=0)  # (seq, seq)
        rows = head_mean[q_start:q_end]  # (n_response, seq_len)
        img_key_mass = rows[:, image_mask].sum(dim=-1)
        attn_sum += img_key_mass
    attn_to_image = (attn_sum / n_layers).detach().float().cpu().numpy()

    target_ids = input_ids[prompt_len:].detach().cpu().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(target_ids)
    return target_ids, tokens, attn_to_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rollout-dir", default="rollouts/Vision-OPD-noimg-Qwen3-VL-2B-Instruct")
    parser.add_argument("--parquet", default="data/train_answer.parquet")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--output-dir", default="analysis_outputs/ra_attention_check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    df = pd.read_parquet(args.parquet)
    row_index = build_row_index(df)
    records = load_rollouts(Path(args.rollout_dir), args.max_samples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    misses = 0
    for sample_id, record in enumerate(tqdm(records, desc="attention")):
        key = (rollout_prompt_text(record["input"]), str(record.get("gts", "")))
        row_idx = row_index.get(key)
        if row_idx is None:
            misses += 1
            continue
        row = df.iloc[row_idx]
        response = str(record.get("output", ""))
        if not response.strip():
            continue
        try:
            messages = make_prompt_messages(row, "images")
        except Exception as e:
            print(f"sample {sample_id} message build failed: {e}")
            continue
        try:
            ids, toks, attn = response_image_attention(model, processor, messages, response, device)
        except Exception as e:
            print(f"sample {sample_id} forward failed: {e}")
            continue
        if len(ids) == 0:
            continue
        for t, (tid, a) in enumerate(zip(ids, attn)):
            text = processor.tokenizer.decode([tid], skip_special_tokens=False)
            rows.append({"sample_id": sample_id, "t": t, "token_id": tid, "text": text, "attn_to_image": float(a)})

    print(f"matched={len(records) - misses} misses={misses} tokens={len(rows)}")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "attn_tokens.csv", index=False)
    print(f"wrote {out_dir / 'attn_tokens.csv'}")


if __name__ == "__main__":
    main()
