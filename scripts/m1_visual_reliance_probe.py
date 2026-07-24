#!/usr/bin/env python3
"""M1 (arXiv plan): motivation probe -- how often does a base VLM give the SAME
answer when the image content is erased?

For each sample in a VLMEvalKit-format TSV (POPE / VStarBench), run greedy
generation twice with identical prompts:
  hi   = real image
  ctrl = same-size black RGB image (placeholder/visual-token count preserved,
         matching the training-time control construction)
and report:
  - unchanged_rate: fraction of samples whose (normalized) answer is identical
  - acc_hi / acc_ctrl: accuracy vs ground truth under each condition
    (black-image accuracy far above chance = language-prior leakage)
  - mean generation length under each condition

This produces the single number cited in the paper intro ("fraction of answers
unchanged under a black control image").

Usage (see experiment_script/run_m1_probe.sh for the wrapper):
  python3 scripts/m1_visual_reliance_probe.py \
    --model-path <base model dir> \
    --tsv <LMUData>/POPE.tsv --dataset-type yesno \
    --num-samples 500 --output analysis_outputs/m1_probe/pope_2b.json
"""

import argparse
import ast
import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

MCQ_SUFFIX = "\nAnswer with the option's letter from the given choices directly."
YESNO_SUFFIX = "\nAnswer the question using a single word or phrase."
OPTION_LETTERS = "ABCDEFGH"


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return not pd.isna(value) and str(value).strip() != ""


def scalar_or_first(value: Any) -> Any:
    """VLMEvalKit image columns are usually scalar, but some datasets store one-item lists."""
    if not is_present(value):
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    text = str(value).strip()
    if len(text) >= 2 and text[0] in "[(" and text[-1] in "])":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, (list, tuple)) and parsed:
            return parsed[0]
    return value


def decode_base64_image(value: Any) -> Image.Image | None:
    value = scalar_or_first(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("data:image") and "," in text:
        text = text.split(",", 1)[1]
    if len(text) < 64:
        return None
    try:
        decoded = base64.b64decode(text, validate=True)
    except Exception:  # noqa: BLE001
        return None
    try:
        with Image.open(io.BytesIO(decoded)) as image:
            return image.convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def load_image(row: pd.Series, tsv_dir: Path) -> Image.Image:
    for column in ("image", "image_path"):
        if column not in row or not is_present(row[column]):
            continue
        image = decode_base64_image(row[column])
        if image is not None:
            return image

        raw_path = scalar_or_first(row[column])
        if isinstance(raw_path, str):
            path = Path(raw_path)
            if not path.is_absolute():
                path = tsv_dir / path
            if path.exists():
                with Image.open(path) as image:
                    return image.convert("RGB")
    raise ValueError("row has neither decodable base64 image nor readable image path")


def build_question(row, dataset_type: str) -> str:
    q = str(row["question"])
    if dataset_type == "mcq":
        opts = []
        for letter in OPTION_LETTERS:
            if letter in row and is_present(row[letter]):
                opts.append(f"{letter}. {str(row[letter]).strip()}")
        if opts:
            q = q + "\n" + "\n".join(opts)
        return q + MCQ_SUFFIX
    return q + YESNO_SUFFIX


def normalize_free_text(text: Any, max_chars: int = 64) -> str:
    if not is_present(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())[:max_chars]


def extract_yesno(text: Any) -> str:
    t = normalize_free_text(text, max_chars=256)
    if not t:
        return ""
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1].strip()
    t = re.sub(r"^<answer>\s*|\s*</answer>$", "", t, flags=re.IGNORECASE).strip()

    patterns = [
        r"^(?:answer\s*(?:is|:)?\s*)?(yes|no)(?:[\s.,;:!?)]|$)",
        r"\b(?:final|correct)\s+answer\s*(?:is|:)?\s*(yes|no)\b",
        r"\banswer\s*(?:is|:)\s*(yes|no)\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, t, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()

    yes = bool(re.search(r"\byes\b", t))
    no = bool(re.search(r"\bno\b", t))
    if yes != no:
        return "yes" if yes else "no"
    return t[:32]


def extract_mcq_option(text: Any, valid_letters: str = OPTION_LETTERS) -> str:
    t = "" if not is_present(text) else str(text).strip()
    if not t:
        return ""
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1].strip()
    answer_span = t
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", t, flags=re.IGNORECASE | re.DOTALL)
    if match:
        answer_span = match.group(1).strip()

    letters = re.escape(valid_letters)
    lower_letters = letters.lower()
    pattern_specs = [
        (
            rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*\*?\*?\s*[\(\[]?\s*([{letters}])"
            r"(?:\s*[\)\].:\-\*]|(?:\s+|$))",
            0,
        ),
        (
            rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*\*?\*?\s*[\(\[]?\s*([{lower_letters}])"
            r"(?:\s*[\)\].:\-\*]|$)",
            0,
        ),
        (rf"\b(?:option|choice|letter)\s*[\(\[]?\s*([{letters}])\b", re.IGNORECASE),
        (rf"^\s*[\(\[]?\s*([{letters}])(?:\s*[\)\].:\-]|(?:\s+|$))", re.IGNORECASE),
        (rf"(?:^|[\n\r])\s*[\(\[]?\s*([{letters}])(?:\s*[\)\].:\-]|(?:\s+|$))", re.IGNORECASE),
    ]
    for pattern, flags in pattern_specs:
        matches = list(re.finditer(pattern, answer_span, flags=flags))
        if matches:
            return matches[-1].group(1).lower()
    return ""


def normalize_ground_truth(value: Any, dataset_type: str, row: pd.Series) -> str:
    if dataset_type == "yesno":
        return extract_yesno(value)
    if dataset_type == "mcq":
        option = extract_mcq_option(value)
        if option:
            return option
        gt_text = normalize_free_text(value, max_chars=256)
        for letter in OPTION_LETTERS:
            if (
                letter in row
                and is_present(row[letter])
                and gt_text == normalize_free_text(row[letter], max_chars=256)
            ):
                return letter.lower()
        return gt_text[:32]
    return normalize_free_text(value)


def normalize_answer(text: str, dataset_type: str) -> str:
    if dataset_type == "yesno":
        return extract_yesno(text)
    if dataset_type == "mcq":
        option = extract_mcq_option(text)
        return option if option else normalize_free_text(text, max_chars=32)
    return normalize_free_text(text)


@torch.no_grad()
def generate(model, processor, image, question, device, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True), int(gen.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tsv", required=True, help="VLMEvalKit-format TSV (POPE.tsv / VStarBench.tsv)")
    ap.add_argument("--dataset-type", choices=["yesno", "mcq", "open"], required=True)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    tsv_path = Path(args.tsv)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    df = pd.read_csv(tsv_path, sep="\t")
    required = {"question", "answer", "image"} if "image" in df.columns else {
        "question",
        "answer",
        "image_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{args.tsv} is missing required columns: {missing}")
    if args.num_samples and args.num_samples < len(df):
        df = df.sample(n=args.num_samples, random_state=args.sample_seed).reset_index(drop=True)

    records, unchanged = [], 0
    acc = {"hi": 0, "ctrl": 0}
    lens = {"hi": [], "ctrl": []}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="probe"):
        try:
            img = load_image(row, tsv_path.parent)
        except Exception as e:  # noqa: BLE001
            print(f"skip row: {e}")
            continue
        question = build_question(row, args.dataset_type)
        black = Image.new("RGB", img.size, (0, 0, 0))

        ans = {}
        for cond, im in (("hi", img), ("ctrl", black)):
            raw, gen_len = generate(model, processor, im, question, device, args.max_new_tokens)
            ans[cond] = normalize_answer(raw, args.dataset_type)
            lens[cond].append(gen_len)
        gt = normalize_ground_truth(row.get("answer", ""), args.dataset_type, row)
        same = ans["hi"] == ans["ctrl"]
        unchanged += int(same)
        for cond in ("hi", "ctrl"):
            acc[cond] += int(ans[cond] == gt and gt != "")
        records.append({"index": int(row.get("index", -1)), "hi": ans["hi"],
                        "ctrl": ans["ctrl"], "gt": gt, "unchanged": same})

    n = len(records)
    summary = {
        "model": args.model_path, "tsv": args.tsv, "dataset_type": args.dataset_type,
        "n": n,
        "unchanged_rate": unchanged / max(n, 1),
        "acc_hi": acc["hi"] / max(n, 1),
        "acc_ctrl": acc["ctrl"] / max(n, 1),
        "mean_len_hi": sum(lens["hi"]) / max(n, 1),
        "mean_len_ctrl": sum(lens["ctrl"]) / max(n, 1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
