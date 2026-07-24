#!/usr/bin/env python3
"""Offline RA-VAD positive/negative token analysis.

This script replays existing rollout responses, reconstructs the high-info and
control teacher prompts used by RA-VAD, and computes per-response-token
``ra_raw = logp_hi - logp_ctrl``.
"""

import argparse
import html
import json
import math
import os
import re
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_PATH = (
    "/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/"
    "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
)

MODE_TO_ROLLOUT_DIR = {
    "qvis": "rollouts/Vision-OPD-qvis-Qwen3-VL-2B-Instruct",
    "qvis_mask": "rollouts/Vision-OPD-qvis-Qwen3-VL-2B-Instruct",
    "iq_vs_i": "rollouts/Vision-OPD-qvis-Qwen3-VL-2B-Instruct",
    "noimg": "rollouts/Vision-OPD-noimg-Qwen3-VL-2B-Instruct",
    "black": "rollouts/Vision-OPD-black-Qwen3-VL-2B-Instruct",
    "degrade": "rollouts/Vision-OPD-degrade-Qwen3-VL-2B-Instruct",
    "answer_hint": "rollouts/Vision-OPD-baseline-Qwen3-VL-2B-Instruct",
}

DEFAULT_ANSWER_HINT_TEMPLATE = (
    "\n\nHere is a reference solution to this problem:\n"
    "{answer}\n\n"
    "After understanding the reference solution, please try to solve this problem using your own approach below:\n"
)

MASK_KEEP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "between",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "from",
    "has",
    "have",
    "how",
    "in",
    "inside",
    "is",
    "it",
    "its",
    "near",
    "next",
    "not",
    "of",
    "on",
    "or",
    "than",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def normalize_text(text: str) -> str:
    text = text.replace("<image>", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rollout_prompt_text(input_text: str) -> str:
    text = input_text
    if text.startswith("user\n\n"):
        text = text[len("user\n\n"):]
    marker = "\nassistant\n"
    if marker in text:
        text = text.split(marker, 1)[0]
    return normalize_text(text)


def parquet_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, list) and prompt:
        content = prompt[-1].get("content", "")
    elif isinstance(prompt, dict):
        content = prompt.get("content", "")
    else:
        content = str(prompt)
    return normalize_text(content)


def entry_to_path(entry: Any) -> str:
    if isinstance(entry, np.ndarray):
        entry = entry.tolist()
    if isinstance(entry, (list, tuple)):
        if not entry:
            raise ValueError("empty image entry list")
        entry = entry[0]
    if isinstance(entry, dict):
        if "path" in entry:
            return str(entry["path"])
        if "image" in entry:
            return entry_to_path(entry["image"])
    if isinstance(entry, str):
        return entry
    raise TypeError(f"unsupported image entry: {type(entry)}")


def load_image(entry: Any) -> Image.Image:
    if isinstance(entry, np.ndarray):
        entry = entry.tolist()
    if isinstance(entry, (list, tuple)):
        if len(entry) != 1:
            return [load_image(x) for x in entry]
        entry = entry[0]
    if isinstance(entry, Image.Image):
        return entry.convert("RGB")
    if isinstance(entry, dict):
        if "image" in entry:
            return load_image(entry["image"])
        if "bytes" in entry:
            return Image.open(BytesIO(entry["bytes"])).convert("RGB")
        if "path" in entry:
            return Image.open(entry["path"]).convert("RGB")
    if isinstance(entry, str):
        return Image.open(entry).convert("RGB")
    raise TypeError(f"unsupported image entry: {type(entry)}")


def list_images(entry: Any) -> list[Image.Image]:
    if isinstance(entry, np.ndarray):
        entry = entry.tolist()
    if isinstance(entry, (list, tuple)):
        return [load_image(x) for x in entry]
    return [load_image(entry)]


def make_prompt_messages(row: pd.Series, image_key: str = "images") -> list[dict[str, Any]]:
    prompt = row["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    messages = deepcopy(prompt)
    images = list_images(row[image_key])
    image_offset = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            continue
        parts = []
        for segment in [x for x in re.split(r"(<image>)", content) if x != ""]:
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError("not enough images for prompt placeholders")
                parts.append({"type": "image", "image": images[image_offset]})
                image_offset += 1
            else:
                parts.append({"type": "text", "text": segment})
        message["content"] = parts
    if image_offset != len(images):
        raise ValueError("image count does not match prompt placeholders")
    return messages


def swap_images(messages: list[dict[str, Any]], images: list[Image.Image]) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    image_offset = 0
    for message in out:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                if image_offset >= len(images):
                    raise ValueError("not enough replacement images")
                new_content.append({"type": "image", "image": images[image_offset]})
                image_offset += 1
            else:
                new_content.append(item)
        message["content"] = new_content
    if image_offset != len(images):
        raise ValueError("replacement image count does not match prompt")
    return out


def remove_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                item
                for item in content
                if not (isinstance(item, dict) and item.get("type") in {"image", "video"})
            ]
    return out


def generic_visual_messages(messages: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                item
                for item in content
                if isinstance(item, dict) and item.get("type") in {"image", "video"}
            ]
    if not out:
        return [{"role": "user", "content": prompt}]
    content = out[-1].get("content")
    if isinstance(content, list):
        out[-1]["content"] = content + [{"type": "text", "text": prompt}]
    else:
        out[-1]["content"] = prompt
    return out


def image_only_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                item
                for item in content
                if isinstance(item, dict) and item.get("type") in {"image", "video"}
            ]
    return out


def mask_content_words(text: str) -> str:
    def mask_line(line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return line
        if re.match(r"^[A-D]\s*[.)]", stripped):
            return line
        lower = stripped.lower()
        if "answer with" in lower or "given choices" in lower:
            return line
        if "bounding box" in lower or "red bounding box" in lower:
            return line

        def repl(match: re.Match[str]) -> str:
            word = match.group(0)
            if word.lower() in MASK_KEEP_WORDS or len(word) <= 2:
                return word
            return "[MASK]"

        return re.sub(r"[A-Za-z]+", repl, line)

    return "\n".join(mask_line(line) for line in text.splitlines())


def masked_question_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                {**item, "text": mask_content_words(str(item.get("text", "")))}
                if isinstance(item, dict) and item.get("type") == "text"
                else item
                for item in content
            ]
        elif isinstance(content, str):
            message["content"] = mask_content_words(content)
    return out


def black_images_like(images: list[Image.Image]) -> list[Image.Image]:
    return [Image.new("RGB", image.size, color=(0, 0, 0)) for image in images]


def add_answer_hint(
    messages: list[dict[str, Any]],
    answer: str,
    hint_template: str,
) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    suffix = hint_template.format(answer=answer)
    content = out[-1].get("content")
    if isinstance(content, list):
        out[-1]["content"] = list(content) + [{"type": "text", "text": suffix}]
    elif isinstance(content, str):
        out[-1]["content"] = content + suffix
    else:
        raise TypeError(f"unsupported message content type: {type(content)}")
    return out


def row_answer(row: pd.Series) -> str:
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
        return str(extra_info["answer"])
    return ""


def build_mode_messages(
    row: pd.Series,
    mode: str,
    answer_hint_template: str = DEFAULT_ANSWER_HINT_TEMPLATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    high_messages = make_prompt_messages(row, "images")
    if mode == "noimg":
        ctrl_messages = remove_images(high_messages)
    elif mode == "black":
        ctrl_messages = swap_images(high_messages, black_images_like(list_images(row["images"])))
    elif mode == "degrade":
        if "images_degraded" not in row or row["images_degraded"] is None:
            raise ValueError("degrade mode requires images_degraded column")
        ctrl_messages = swap_images(high_messages, list_images(row["images_degraded"]))
    elif mode == "qvis":
        ctrl_messages = generic_visual_messages(high_messages, "Describe this image in detail.")
    elif mode == "qvis_mask":
        ctrl_messages = masked_question_messages(high_messages)
    elif mode == "iq_vs_i":
        ctrl_messages = image_only_messages(high_messages)
    elif mode == "answer_hint":
        ctrl_messages = add_answer_hint(high_messages, row_answer(row), answer_hint_template)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return high_messages, ctrl_messages


def extract_images_for_processor(messages: list[dict[str, Any]]) -> list[Image.Image] | None:
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                images.append(load_image(item["image"]))
    return images or None


def text_for_processor(processor, messages: list[dict[str, Any]], response: str | None) -> str:
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if response is None:
        return prompt
    return prompt + response


@torch.no_grad()
def response_logprobs(
    model,
    processor,
    messages: list[dict[str, Any]],
    response: str,
    device: torch.device,
) -> tuple[list[int], list[str], torch.Tensor]:
    images = extract_images_for_processor(messages)
    prompt_text = text_for_processor(processor, messages, response=None)
    full_text = text_for_processor(processor, messages, response=response)

    prompt_inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=images, return_tensors="pt")
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    full_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_inputs.items()}
    outputs = model(**full_inputs)
    input_ids = full_inputs["input_ids"][0]
    if prompt_len >= input_ids.numel():
        return [], [], torch.empty(0, device="cpu")

    target_ids = input_ids[prompt_len:]
    logits = outputs.logits[0, prompt_len - 1 : input_ids.numel() - 1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1).detach().cpu()
    token_ids = target_ids.detach().cpu().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    return token_ids, tokens, token_logps


def load_rollouts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    records = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    return records
    return records


def build_row_index(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    mapping = {}
    for idx, row in df.iterrows():
        answer = str(row.get("reward_model", {}).get("ground_truth", row.get("extra_info", {}).get("answer", "")))
        key = (parquet_prompt_text(row["prompt"]), answer)
        if key not in mapping:
            mapping[key] = int(idx)
    return mapping


def token_kind(value: float, eps: float) -> str:
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "near_zero"


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def compute_ra_weight_arrays(
    ra_raw: np.ndarray,
    *,
    delta: float,
    clip_quantile: float,
    min_positive_tokens: int,
    no_sample_gate: bool,
    margin_scale: float,
    answer_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Numpy mirror of verl.trainer.ppo.ra_vad.compute_ra_weights for one sample."""
    ra_pos = np.maximum(ra_raw - float(delta), 0.0)
    positive_mask = ra_pos > 0
    positive_count = int(positive_mask.sum())

    if positive_count <= 0 or positive_count < min_positive_tokens:
        return ra_pos, np.zeros_like(ra_raw), np.zeros_like(ra_raw), 0.0

    clipped = ra_pos.copy()
    if clip_quantile < 1.0:
        positive_values = clipped[positive_mask]
        # Match the torch implementation approximately: ceil(count*q)-1 over
        # sorted positive values.
        quantile_idx = max(0, min(positive_count - 1, math.ceil(positive_count * clip_quantile) - 1))
        clip_value = np.sort(positive_values)[quantile_idx]
        clipped = np.minimum(clipped, clip_value)

    positive_mean = clipped[positive_mask].mean() if positive_count > 0 else 0.0
    if positive_mean <= 0:
        return ra_pos, np.zeros_like(ra_raw), np.zeros_like(ra_raw), 0.0
    ra_norm = clipped / positive_mean

    if no_sample_gate:
        g_sample = 1.0
    else:
        margin = float(ra_raw.mean()) if ra_raw.size else 0.0
        ra_answer = float(ra_norm[positive_mask].mean()) if positive_count > 0 else 0.0
        g_sample = sigmoid(margin / float(margin_scale)) * sigmoid(ra_answer / float(answer_scale))

    weights = ra_norm * g_sample
    return ra_pos, ra_norm, weights, float(g_sample)


def color_for_value(value: float, max_abs: float) -> str:
    if max_abs <= 0 or math.isnan(value):
        return "rgba(160,160,160,0.12)"
    alpha = min(0.85, 0.12 + 0.73 * abs(value) / max_abs)
    if value > 0:
        return f"rgba(210, 45, 45, {alpha:.3f})"
    if value < 0:
        return f"rgba(45, 90, 210, {alpha:.3f})"
    return "rgba(160,160,160,0.12)"


def render_html(samples: list[dict[str, Any]], out_path: Path, max_html_samples: int) -> None:
    rows = []
    for sample in samples[:max_html_samples]:
        token_html = []
        values = [abs(tok["ra_raw"]) for tok in sample["tokens"]]
        max_abs = max(values) if values else 0.0
        for tok in sample["tokens"]:
            text = html.escape(tok["text"].replace("\n", "\\n"))
            title = (
                f"ra_raw={tok['ra_raw']:.4f}, "
                f"logp_hi={tok['logp_hi']:.4f}, logp_ctrl={tok['logp_ctrl']:.4f}, kind={tok['kind']}"
            )
            token_html.append(
                f'<span class="tok" title="{html.escape(title)}" '
                f'style="background:{color_for_value(tok["ra_raw"], max_abs)}">{text}</span>'
            )
        rows.append(
            "<section>"
            f"<h3>sample {sample['sample_id']} | mode={html.escape(sample['mode'])} | "
            f"gt={html.escape(str(sample.get('gt', '')))}</h3>"
            f"<p><b>Question:</b> {html.escape(sample['question'])}</p>"
            f"<p><b>Summary:</b> pos={sample['positive_ratio']:.3f}, "
            f"neg={sample['negative_ratio']:.3f}, pos_mass={sample['positive_mass']:.3f}, "
            f"neg_mass={sample['negative_mass']:.3f}</p>"
            f'<div class="tokens">{"".join(token_html)}</div>'
            "</section>"
        )
    page = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
body { font-family: system-ui, sans-serif; margin: 24px; line-height: 1.45; }
section { border-bottom: 1px solid #ddd; padding: 18px 0; }
.tokens { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }
.tok { border-radius: 3px; padding: 1px 2px; margin: 0 1px; }
</style></head><body>
<h1>RA token attribution</h1>
<p>Red = positive token (high-info &gt; control), blue = negative token (control &gt; high-info).</p>
""" + "\n".join(rows) + "\n</body></html>\n"
    out_path.write_text(page, encoding="utf-8")


def analyze_mode(args, mode: str, model, processor, device: torch.device) -> list[dict[str, Any]]:
    parquet_path = Path(args.degraded_parquet if mode == "degrade" else args.parquet)
    df = pd.read_parquet(parquet_path)
    row_index = build_row_index(df)

    rollout_dir = Path(args.rollout_dir or MODE_TO_ROLLOUT_DIR[mode])
    records = load_rollouts(rollout_dir, args.max_samples)
    out_dir = Path(args.output_dir) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_summaries = []
    token_rows = []
    misses = 0
    for sample_id, record in enumerate(tqdm(records, desc=f"analyze {mode}", unit="sample")):
        key = (rollout_prompt_text(record["input"]), str(record.get("gts", "")))
        row_idx = row_index.get(key)
        if row_idx is None:
            misses += 1
            continue
        row = df.iloc[row_idx]
        response = str(record.get("output", ""))
        if not response.strip():
            continue
        high_messages, ctrl_messages = build_mode_messages(row, mode, args.answer_hint_template)
        ids_hi, toks_hi, logp_hi = response_logprobs(model, processor, high_messages, response, device)
        ids_ctrl, toks_ctrl, logp_ctrl = response_logprobs(model, processor, ctrl_messages, response, device)
        n = min(len(ids_hi), len(ids_ctrl))
        if n == 0:
            continue
        if ids_hi[:n] != ids_ctrl[:n]:
            # Different prompt conditions can very rarely tokenize response boundaries
            # differently. Keep the shared prefix only.
            shared = 0
            for a, b in zip(ids_hi, ids_ctrl):
                if a != b:
                    break
                shared += 1
            n = shared
        if n == 0:
            continue
        ra_raw = (logp_hi[:n] - logp_ctrl[:n]).numpy()
        ra_pos, ra_norm, ra_weight, g_sample = compute_ra_weight_arrays(
            ra_raw,
            delta=args.ra_delta,
            clip_quantile=args.ra_clip_quantile,
            min_positive_tokens=args.ra_min_positive_tokens,
            no_sample_gate=args.ra_no_sample_gate,
            margin_scale=args.ra_margin_scale,
            answer_scale=args.ra_answer_scale,
        )
        pos = ra_raw > args.eps
        neg = ra_raw < -args.eps
        positive_mass = float(np.maximum(ra_raw, 0).sum())
        negative_mass = float(np.maximum(-ra_raw, 0).sum())

        tokens = []
        for t, token_id in enumerate(ids_hi[:n]):
            text = processor.tokenizer.decode([token_id], skip_special_tokens=False)
            kind = token_kind(float(ra_raw[t]), args.eps)
            tok = {
                "t": t,
                "token_id": int(token_id),
                "token": toks_hi[t],
                "text": text,
                "logp_hi": float(logp_hi[t]),
                "logp_ctrl": float(logp_ctrl[t]),
                "ra_raw": float(ra_raw[t]),
                "ra_pos": float(ra_pos[t]),
                "ra_norm": float(ra_norm[t]),
                "ra_weight": float(ra_weight[t]),
                "kind": kind,
            }
            tokens.append(tok)
            token_rows.append(
                {
                    "mode": mode,
                    "sample_id": sample_id,
                    "row_idx": row_idx,
                    "gt": record.get("gts"),
                    **tok,
                }
            )

        question = row.get("extra_info", {}).get("question", rollout_prompt_text(record["input"]))
        sample_summaries.append(
            {
                "mode": mode,
                "sample_id": sample_id,
                "row_idx": row_idx,
                "gt": record.get("gts"),
                "question": str(question),
                "response": response,
                "num_tokens": int(n),
                "positive_ratio": float(pos.mean()),
                "negative_ratio": float(neg.mean()),
                "near_zero_ratio": float((~pos & ~neg).mean()),
                "positive_mass": positive_mass,
                "negative_mass": negative_mass,
                "weight_mass": float(ra_weight.sum()),
                "weight_mean": float(ra_weight.mean()),
                "g_sample": g_sample,
                "mean_ra_raw": float(ra_raw.mean()),
                "tokens": tokens,
            }
        )

    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for sample in sample_summaries:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    pd.DataFrame(token_rows).to_csv(out_dir / "tokens.csv", index=False)
    summary_df = pd.DataFrame(
        [
            {k: v for k, v in sample.items() if k not in {"tokens", "response"}}
            for sample in sample_summaries
        ]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    render_html(sample_summaries, out_dir / "token_attribution.html", args.max_html_samples)

    print(f"[{mode}] matched={len(sample_summaries)} missed={misses} out={out_dir}")
    return sample_summaries


def write_paired_comparison(out_root: Path, modes: list[str]) -> None:
    frames = []
    for mode in modes:
        token_path = out_root / mode / "tokens.csv"
        if not token_path.exists():
            continue
        df = pd.read_csv(token_path)
        keep = ["sample_id", "t", "token_id", "text", "ra_raw", "ra_weight", "kind"]
        df = df[keep].rename(
            columns={
                "ra_raw": f"{mode}_ra_raw",
                "ra_weight": f"{mode}_ra_weight",
                "kind": f"{mode}_kind",
            }
        )
        frames.append((mode, df))
    if len(frames) < 2:
        return

    merged = frames[0][1]
    for _, df in frames[1:]:
        merged = merged.merge(df, on=["sample_id", "t", "token_id", "text"], how="inner")
    merged.to_csv(out_root / "paired_tokens_wide.csv", index=False)

    rows = []
    raw_cols = [f"{mode}_ra_raw" for mode, _ in frames if f"{mode}_ra_raw" in merged]
    weight_cols = [f"{mode}_ra_weight" for mode, _ in frames if f"{mode}_ra_weight" in merged]
    for metric, cols in [("ra_raw", raw_cols), ("ra_weight", weight_cols)]:
        corr = merged[cols].corr()
        corr.to_csv(out_root / f"paired_{metric}_corr.csv")
        for i, left in enumerate(cols):
            for right in cols[i + 1 :]:
                diff = merged[left] - merged[right]
                rows.append(
                    {
                        "metric": metric,
                        "left": left.replace(f"_{metric}", ""),
                        "right": right.replace(f"_{metric}", ""),
                        "tokens": len(diff),
                        "corr": float(merged[left].corr(merged[right])),
                        "mean_abs_diff": float(diff.abs().mean()),
                        "mean_signed_diff": float(diff.mean()),
                        "left_gt_right_ratio": float((diff > 0).mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(out_root / "paired_mode_differences.csv", index=False)

    sign_rows = []
    for i, (left_mode, _) in enumerate(frames):
        left_kind = f"{left_mode}_kind"
        if left_kind not in merged:
            continue
        for right_mode, _ in frames[i + 1 :]:
            right_kind = f"{right_mode}_kind"
            if right_kind not in merged:
                continue
            left_pos = merged[left_kind] == "positive"
            right_pos = merged[right_kind] == "positive"
            union = left_pos | right_pos
            non_zero = (merged[left_kind] != "near_zero") & (merged[right_kind] != "near_zero")
            same_sign = non_zero & (merged[left_kind] == merged[right_kind])
            opposite_sign = non_zero & (merged[left_kind] != merged[right_kind])
            sign_rows.append(
                {
                    "left": left_mode,
                    "right": right_mode,
                    "tokens": int(len(merged)),
                    "positive_jaccard": float((left_pos & right_pos).sum() / max(1, union.sum())),
                    "same_sign_ratio": float(same_sign.sum() / max(1, non_zero.sum())),
                    "opposite_sign_ratio": float(opposite_sign.sum() / max(1, non_zero.sum())),
                    "non_zero_tokens": int(non_zero.sum()),
                }
            )
    pd.DataFrame(sign_rows).to_csv(out_root / "paired_sign_jaccard.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze positive/negative RA-VAD tokens by control mode.")
    parser.add_argument("--modes", default="qvis,noimg,black,degrade", help="Comma-separated modes.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--parquet", default="data/train_answer.parquet")
    parser.add_argument("--degraded-parquet", default="data/train_degraded_answer.parquet")
    parser.add_argument("--rollout-dir", default=None, help="Override rollout dir for all requested modes.")
    parser.add_argument("--output-dir", default="analysis_outputs/ra_tokens")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-html-samples", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--ra-delta", type=float, default=0.0)
    parser.add_argument("--ra-clip-quantile", type=float, default=0.95)
    parser.add_argument("--ra-min-positive-tokens", type=int, default=1)
    parser.add_argument("--ra-no-sample-gate", action="store_true")
    parser.add_argument("--ra-margin-scale", type=float, default=0.5)
    parser.add_argument("--ra-answer-scale", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--answer-hint-template",
        default=DEFAULT_ANSWER_HINT_TEMPLATE,
        help="Template used by the answer_hint diagnostic control mode. Must include {answer}.",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.eval()

    all_rows = []
    for mode in modes:
        summaries = analyze_mode(args, mode, model, processor, device)
        if summaries:
            all_rows.append(
                {
                    "mode": mode,
                    "samples": len(summaries),
                    "tokens": sum(s["num_tokens"] for s in summaries),
                    "positive_ratio": np.average(
                        [s["positive_ratio"] for s in summaries],
                        weights=[s["num_tokens"] for s in summaries],
                    ),
                    "negative_ratio": np.average(
                        [s["negative_ratio"] for s in summaries],
                        weights=[s["num_tokens"] for s in summaries],
                    ),
                    "positive_mass_per_token": sum(s["positive_mass"] for s in summaries)
                    / max(1, sum(s["num_tokens"] for s in summaries)),
                    "negative_mass_per_token": sum(s["negative_mass"] for s in summaries)
                    / max(1, sum(s["num_tokens"] for s in summaries)),
                    "mean_ra_raw": np.average(
                        [s["mean_ra_raw"] for s in summaries],
                        weights=[s["num_tokens"] for s in summaries],
                    ),
                }
            )
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_root / "compare_modes.csv", index=False)
    write_paired_comparison(out_root, modes)
    print(f"Wrote mode comparison to {out_root / 'compare_modes.csv'}")


if __name__ == "__main__":
    main()
