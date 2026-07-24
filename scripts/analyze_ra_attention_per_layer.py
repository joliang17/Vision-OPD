#!/usr/bin/env python3
"""Per-layer variant of analyze_ra_attention.py: instead of averaging attention-to-image
concentration across a fixed set of trailing layers, keep every layer's score separately so we
can see which individual layers actually carry the "attends to true visual content, not
discourse markers" signal found in the averaged version (see docs/compare_vaopd_0701.md Phase
2-核心 第七/八轮) — i.e. answers "how do we know layer N is a good choice for
ra_attention_num_layers" instead of just asserting it.
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

DISCOURSE_WORDS = {"user", "based", "provided", "analysis", "let", "determine"}
CONTENT_WORDS = {
    "wall", "building", "person", "tower", "wooden", "window", "bicycle", "bench",
    "sculpture", "roof", "small", "large", "bright", "camera", "cow", "barrier",
    "red", "blue", "green", "yellow", "white", "black", "pink", "orange", "purple", "brown",
}


@torch.no_grad()
def response_image_attention_per_layer(model, processor, messages, response, device):
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

    image_mask = (input_ids == IMAGE_TOKEN_ID).cpu().numpy()
    if image_mask.sum() == 0:
        return None, None

    outputs = model.model(**full_inputs, output_attentions=True, use_cache=False)
    n_layers = len(outputs.attentions)
    seq_len = input_ids.numel()

    q_start, q_end = prompt_len - 1, seq_len - 1
    if q_end <= q_start:
        return None, None

    per_layer_scores = np.zeros((n_layers, q_end - q_start), dtype=np.float32)
    for li, layer_attn in enumerate(outputs.attentions):
        head_mean = layer_attn[0].mean(dim=0)  # (seq, seq)
        rows = head_mean[q_start:q_end]  # (n_response, seq_len)
        img_key_mass = rows[:, image_mask].sum(dim=-1)
        per_layer_scores[li] = img_key_mass.detach().float().cpu().numpy()

    target_ids = input_ids[prompt_len:].detach().cpu().tolist()
    tokens = processor.tokenizer.decode(target_ids, skip_special_tokens=False)
    token_texts = [processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in target_ids]
    return per_layer_scores, token_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rollout-dir", default="rollouts/Vision-OPD-noimg-Qwen3-VL-2B-Instruct")
    parser.add_argument("--parquet", default="data/train_answer.parquet")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--output-dir", default="analysis_outputs/ra_attention_per_layer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
    ).to(device)
    model.eval()

    df = pd.read_parquet(args.parquet)
    row_index = build_row_index(df)
    records = load_rollouts(Path(args.rollout_dir), args.max_samples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = None
    layer_content_sum = None
    layer_content_n = 0
    layer_discourse_sum = None
    layer_discourse_n = 0
    layer_overall_sum = None
    layer_overall_n = 0

    misses = 0
    for sample_id, record in enumerate(tqdm(records, desc="per-layer attention")):
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
        except Exception:
            continue
        try:
            scores, token_texts = response_image_attention_per_layer(model, processor, messages, response, device)
        except Exception as e:
            print(f"sample {sample_id} failed: {e}")
            continue
        if scores is None:
            continue
        if n_layers is None:
            n_layers = scores.shape[0]
            layer_content_sum = np.zeros(n_layers)
            layer_discourse_sum = np.zeros(n_layers)
            layer_overall_sum = np.zeros(n_layers)

        for t, text in enumerate(token_texts):
            norm = text.strip().lower()
            layer_overall_sum += scores[:, t]
            layer_overall_n += 1
            if norm in CONTENT_WORDS:
                layer_content_sum += scores[:, t]
                layer_content_n += 1
            elif norm in DISCOURSE_WORDS:
                layer_discourse_sum += scores[:, t]
                layer_discourse_n += 1

    print(f"matched={len(records) - misses} misses={misses}")
    print(f"content_n={layer_content_n} discourse_n={layer_discourse_n} overall_n={layer_overall_n}")

    rows = []
    for li in range(n_layers):
        content_mean = layer_content_sum[li] / max(1, layer_content_n)
        discourse_mean = layer_discourse_sum[li] / max(1, layer_discourse_n)
        overall_mean = layer_overall_sum[li] / max(1, layer_overall_n)
        rows.append(
            {
                "layer": li,
                "content_mean": content_mean,
                "discourse_mean": discourse_mean,
                "overall_mean": overall_mean,
                "content_vs_discourse_ratio": content_mean / max(discourse_mean, 1e-8),
                "content_vs_overall_ratio": content_mean / max(overall_mean, 1e-8),
            }
        )
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "per_layer_summary.csv", index=False)
    print(result_df.sort_values("content_vs_discourse_ratio", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
