#!/usr/bin/env python3
"""Merge the sr1-filtered and virl39k-filtered 'mixed' rollout selections
back into a single verl-format training parquet (prompt/images/ability/
reward_model/data_source), pulling the full original rows (with image
bytes) from the source parquets by re-deriving each selected row's original
index from its rollout-selector id (`{tag}:{shard_idx}:{local_idx}`, where
original_index = shard_idx + local_idx * num_shards).
"""
import argparse
import json

import pandas as pd

NUM_SHARDS = 5

SOURCES = {
    "sr1filtered": "data/vision_sr1_47k_noimg_v2_filtered.parquet",
    "virl39kfiltered": "data/virl39k_train_noimg_filtered.parquet",
}
MIXED_FILES = {
    "sr1filtered": "outputs/rollout_mixed/sr1filtered_mixed_selected.jsonl",
    "virl39kfiltered": "outputs/rollout_mixed/virl39kfiltered_mixed_selected.jsonl",
}


def original_index(rec_id: str) -> tuple[str, int]:
    tag, shard_idx, local_idx = rec_id.split(":")
    return tag, int(shard_idx) + int(local_idx) * NUM_SHARDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sr1_virl39k_mixed_selected_training.parquet")
    args = ap.parse_args()

    frames = []
    for tag, mixed_fp in MIXED_FILES.items():
        indices = []
        for line in open(mixed_fp):
            rec = json.loads(line)
            rec_tag, orig_idx = original_index(rec["id"])
            assert rec_tag == tag
            indices.append(orig_idx)

        src_df = pd.read_parquet(SOURCES[tag])
        sel = src_df.iloc[sorted(indices)].copy()
        sel["mixed_source_tag"] = tag
        frames.append(sel)
        print(f"{tag}: selected {len(sel)} / {len(src_df)} rows from {SOURCES[tag]}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    merged.to_parquet(args.out)
    print(f"wrote {len(merged)} rows -> {args.out}")
    print(merged["data_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
