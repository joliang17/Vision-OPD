#!/usr/bin/env python3
"""Score VLMEvalKit MCQ predictions with extraction plus optional LLM judge.

This is intended for cases where model outputs are verbose and VLMEvalKit's
exact matching undercounts answers like "Correct Answer: **D. ...**".
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI


EXTRACT_PATTERNS = [
    re.compile(r"(?:correct|final)\s*(?:answer)?\s*(?:is|:)?\s*\*?\*?\s*([A-D])\b", re.I),
    re.compile(
        r"(?:therefore|answer)\s*,?\s*(?:the\s*)?(?:correct\s*)?(?:answer\s*)?"
        r"(?:is|:)?\s*\*?\*?\s*([A-D])\b",
        re.I,
    ),
    re.compile(r"\b(?:option|choice)\s*([A-D])\b", re.I),
    re.compile(r"\n\s*([A-D])\.\s"),
    re.compile(r"^\s*([A-D])(?:\.|\)|\s|$)", re.I),
    re.compile(r"\*\*\s*([A-D])\s*(?:\.|\)|:|\b)", re.I),
]

FINAL_PATTERNS = [
    re.compile(r"FINAL\s*:\s*([A-DZ])", re.I),
    re.compile(r'"answer"\s*:\s*"([A-DZ])"', re.I),
    re.compile(r"\b([A-DZ])\b"),
]


def extract_option(prediction: object) -> Optional[str]:
    text = "" if pd.isna(prediction) else str(prediction)
    tail = text[-1600:]
    for pattern in EXTRACT_PATTERNS:
        matches = list(pattern.finditer(tail))
        if matches:
            return matches[-1].group(1).upper()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-10:]):
        match = re.match(r"^([A-D])\b", line, re.I)
        if match:
            return match.group(1).upper()
    return None


def parse_judge_response(text: str) -> Optional[str]:
    tail = text[-600:]
    for pattern in FINAL_PATTERNS:
        matches = list(pattern.finditer(tail))
        if matches:
            return matches[-1].group(1).upper()
    return None


def build_prompt(row: pd.Series) -> str:
    options = "\n".join(
        f"{letter}. {row[letter]}"
        for letter in "ABCD"
        if letter in row and not pd.isna(row[letter])
    )
    return (
        "Extract the final option selected in MODEL OUTPUT. Do not answer the image question.\n"
        "If MODEL OUTPUT does not select any option, write FINAL: Z.\n"
        "At the end, write FINAL: <A/B/C/D/Z>.\n\n"
        f"QUESTION:\n{row['question']}\n\n"
        f"OPTIONS:\n{options}\n\n"
        f"MODEL OUTPUT:\n{row['prediction']}\n"
    )


def judge_one(client: OpenAI, model: str, row: pd.Series, max_tokens: int) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(row)}],
        temperature=0,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    return {"judge_text": text, "judge_option": parse_judge_response(text)}


def score_file(
    path: Path,
    dataset: str,
    client: Optional[OpenAI],
    judge_model: Optional[str],
    workers: int,
    max_tokens: int,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(path)
    df["extracted_option"] = df["prediction"].map(extract_option)
    df["judge_text"] = ""
    df["judge_option"] = pd.NA

    unresolved = df[df["extracted_option"].isna()].copy()
    if client is not None and judge_model and len(unresolved):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(judge_one, client, judge_model, row, max_tokens): idx
                for idx, row in unresolved.iterrows()
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:  # Keep partial scoring inspectable.
                    result = {"judge_text": f"ERROR: {exc}", "judge_option": None}
                df.at[idx, "judge_text"] = result["judge_text"]
                df.at[idx, "judge_option"] = result["judge_option"]

    df["scored_option"] = df["extracted_option"].fillna(df["judge_option"])
    df["hit_scored"] = (df["scored_option"] == df["answer"]).astype(int)
    summary = {
        "dataset": dataset,
        "rows": int(len(df)),
        "extract_coverage": int(df["extracted_option"].notna().sum()),
        "judge_attempted": int(df["extracted_option"].isna().sum()),
        "judge_resolved": int(df["judge_option"].notna().sum()),
        "hits": int(df["hit_scored"].sum()),
        "accuracy": float(df["hit_scored"].mean()),
    }
    if "category" in df.columns:
        summary["category_accuracy"] = {
            str(k): float(v)
            for k, v in df.groupby("category")["hit_scored"].mean().sort_index().items()
        }
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--datasets", nargs="+", default=["MMStar", "VStarBench"])
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    client = None
    if args.judge_base_url and args.judge_model:
        client = OpenAI(base_url=args.judge_base_url.rstrip("/"), api_key=args.judge_api_key)

    summaries = []
    for dataset in args.datasets:
        path = args.run_dir / f"{args.model_name}_{dataset}.xlsx"
        scored, summary = score_file(
            path=path,
            dataset=dataset,
            client=client,
            judge_model=args.judge_model,
            workers=args.workers,
            max_tokens=args.max_tokens,
        )
        out_xlsx = args.run_dir / f"{args.model_name}_{dataset}_extract_plus_judge_result.xlsx"
        out_csv = args.run_dir / f"{args.model_name}_{dataset}_extract_plus_judge_acc.csv"
        scored.to_excel(out_xlsx, index=False)
        pd.DataFrame([summary]).drop(columns=["category_accuracy"], errors="ignore").to_csv(out_csv, index=False)
        summaries.append(summary)

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
