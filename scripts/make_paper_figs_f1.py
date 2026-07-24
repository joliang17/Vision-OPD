#!/usr/bin/env python3
"""F1 paper figures, v2 (2026-07-17) — one visual claim per figure.

Fig 1: two per-token distributions on a shared log axis — the teacher's hi-vs-ctrl gap
(what the method uses as a WEIGHT) vs the teacher-student KL (what a distillation TARGET
can actually teach). Three orders of magnitude apart => the informative signal was being
used as a scalar weight while the target carried nothing.

(Fig 2 [loss trajectory] was cut on 2026-07-17 per user: the loss shape is not to be
shown or discussed in the paper.)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 220, "savefig.bbox": "tight",
})
BLUE = "#3B6EA5"; RED = "#C4502E"; GREEN = "#2E7D4F"; GREY = "#9AA3AF"; INK = "#2A3340"

# ================= Fig 1: two distributions at the top-weight tokens =================
df = pd.read_csv(ROOT / "analysis_outputs" / "teacher_student_kl_diag_virl39k464.csv")
q90 = df["ra_weight"].quantile(0.9)
top = df[df["ra_weight"] > q90]
gap = np.abs(top["ra_raw"].to_numpy()); gap = gap[gap > 0]
kl = top["kl"].to_numpy(); kl = kl[kl > 0]

bins = np.logspace(-5.5, 1.2, 80)
def ridge(vals):
    h, e = np.histogram(np.clip(vals, bins[0], bins[-1]), bins=bins)
    return h / h.max(), (e[:-1] + e[1:]) / 2

fig, ax = plt.subplots(figsize=(4.6, 2.7))
for vals, color, label in [
    (kl, BLUE, "teacher\u2013student KL  (what the target can teach)"),
    (gap, RED, "teacher hi-vs-ctrl gap  (what the weight is based on)"),
]:
    h, c = ridge(vals)
    ax.fill_between(c, 0, h, color=color, alpha=0.55, lw=0)
    ax.plot(c, h, color=color, lw=1.4, label=label)
ax.set_xscale("log")
ax.set_ylim(0, 1.42)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("per-token magnitude (nats, log scale) \u2014 top-10% weighted tokens")

med_kl, med_gap = np.median(kl), np.median(gap)
for v, color in [(med_kl, BLUE), (med_gap, RED)]:
    ax.axvline(v, color=color, lw=1.0, ls=(0, (4, 3)), ymax=0.62)
ax.annotate("", xy=(med_gap, 0.93), xytext=(med_kl, 0.93),
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color=INK))
ax.text(np.sqrt(med_kl * med_gap), 0.99, f"$\\sim${med_gap/med_kl:,.0f}$\\times$",
        ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
ax.text(med_kl, 0.68, f"median\n{med_kl:.4f}", ha="center", fontsize=7.5, color=BLUE)
ax.text(med_gap, 0.68, f"median\n{med_gap:.2f}", ha="center", fontsize=7.5, color=RED)
ax.legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0, fontsize=8)
fig.savefig(OUT / "fig_diag_kl_vs_gap.pdf")
fig.savefig(OUT / "fig_diag_kl_vs_gap.png")
print(f"fig1: top-decile median kl={med_kl:.5f} gap={med_gap:.3f} ratio={med_gap/med_kl:.0f}")

