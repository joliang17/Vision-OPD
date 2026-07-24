#!/usr/bin/env python3
"""扫描 VLMEvalKit 评测输出中的 API 失败污染，结果登记到 docs/eval_integrity_registry.md。

背景：2026-07-15 发现 uniform-weight 的 HRBench8K 有 61/800 条预测是
"Failed to obtain answer via API."（vLLM 请求失败的占位串，全部被判错），把分数从 ~71.9 压到 66.38。
本脚本扫所有 normal_scoring/*_normal.xlsx（每次评测送去判分的最终预测文件），统计污染条数。

用法：python3 scripts/scan_eval_integrity.py            # 全量扫描并重写登记文件
不占 GPU，纯读文件。增量跑：已在登记文件里且文件 mtime 没变的条目直接复用。
"""
import glob
import os
import re
import sys
from datetime import datetime

import pandas as pd

REPO = "/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL = os.path.join(REPO, "VLMEvalKit")
REGISTRY = os.path.join(REPO, "Vision-OPD/docs/eval_integrity_registry.md")
FAIL_STR = "Failed to obtain answer via API."

def parse_existing(path):
    """从已有登记文件读回 (file_rel -> (mtime, row_str))，用于增量。"""
    cache = {}
    if not os.path.exists(path):
        return cache
    for line in open(path):
        m = re.match(r"\| `([^`]+)` \| ([0-9.]+) \|", line)
        if m:
            cache[m.group(1)] = (float(m.group(2)), line.rstrip("\n"))
    return cache

def main():
    files = sorted(
        glob.glob(os.path.join(VLMEVAL, "outputs_*", "*", "normal_scoring", "*_normal.xlsx"))
    )
    cache = parse_existing(REGISTRY)
    rows = []
    dirty = []
    for f in files:
        rel = os.path.relpath(f, VLMEVAL)
        mtime = os.path.getmtime(f)
        if rel in cache and abs(cache[rel][0] - mtime) < 1:
            rows.append(cache[rel][1])
            if "| ❌" in cache[rel][1]:
                dirty.append(cache[rel][1])
            continue
        try:
            df = pd.read_excel(f, usecols=lambda c: c in ("prediction",))
            pred = df["prediction"].astype(str)
            n = len(pred)
            n_fail = int((pred == FAIL_STR).sum())
            n_empty = int((pred.str.strip() == "").sum())
        except Exception as e:
            rows.append(f"| `{rel}` | {mtime:.0f} | ? | ? | ? | ⚠️ 读取失败: {type(e).__name__} |")
            continue
        verdict = "✅" if (n_fail == 0 and n_empty == 0) else "❌"
        row = f"| `{rel}` | {mtime:.0f} | {n} | {n_fail} | {n_empty} | {verdict} |"
        rows.append(row)
        if verdict == "❌":
            dirty.append(row)

    with open(REGISTRY, "w") as out:
        out.write(f"""# 评测完整性登记 — API 失败污染扫描

**这个文件是全部机器共享的评测数字可信度登记表。** 引用任何评测数字前先来这里查一眼对应文件是不是 ✅。
由 `Vision-OPD/scripts/scan_eval_integrity.py` 生成/增量更新（纯读文件不占GPU），最后更新：{datetime.now().strftime("%Y-%m-%d %H:%M")}。

**判定标准**：预测列中出现 `"{FAIL_STR}"`（vLLM 请求失败占位串，会被 judge 全部判错、系统性压低分数）
或空预测即为 ❌ 污染；❌ 的数字不可引用，需要重跑推理（不是重跑 judge——回答本身没生成出来）。
已知案例：uniform-weight HRBench8K 61/800 失败，66.38 → 重跑中（预估 ~71.9）。

## ❌ 被污染的评测（重点看这里，共 {len(dirty)} 个）

| 文件 | mtime | 总条数 | API失败 | 空预测 | 判定 |
|---|---|---|---|---|---|
""")
        for r in dirty:
            out.write(r + "\n")
        out.write(f"""
## 全部扫描结果（共 {len(rows)} 个文件）

| 文件 | mtime | 总条数 | API失败 | 空预测 | 判定 |
|---|---|---|---|---|---|
""")
        for r in rows:
            out.write(r + "\n")

    print(f"扫描完成: {len(rows)} 个文件, {len(dirty)} 个污染")
    for r in dirty:
        print(r)

if __name__ == "__main__":
    sys.exit(main())
