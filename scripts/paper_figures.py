#!/usr/bin/env python3
"""Paper figures for CS-OPSD (Qwen3-VL-2B ablations).
Reusable: edit the DATA blocks and re-run. Outputs PDFs into overleaf-paper/figures/.
seaborn is unavailable in the conda env, so we replicate a seaborn-whitegrid look
with matplotlib rcParams. Run:
    source .../miniconda3/etc/profile.d/conda.sh && conda activate qwen35
    python3 Vision-OPD/scripts/paper_figures.py
All numbers = new 7-bench suite (HR Average, HallusionBench three-mean), 2026-07-22.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/overleaf-paper/figures"
os.makedirs(OUT, exist_ok=True)

# ---- seaborn-whitegrid-like style ----
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 13,
    "font.family": "DejaVu Sans", "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.grid": True, "grid.color": "#CCCCCC", "grid.linewidth": 0.8, "grid.alpha": 0.6,
    "axes.edgecolor": "#444444", "axes.linewidth": 1.0, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
    "xtick.direction": "out", "ytick.direction": "out",
})
C_OURS = "#2E5E8C"   # blue
C_ALT  = "#B4452F"   # red/orange
C_3RD  = "#2E7D4F"   # green


def _smooth(x, y, n=200):
    """Monotone-ish smoothing via cubic spline for visual continuity."""
    try:
        from scipy.interpolate import make_interp_spline
        xs = np.linspace(min(x), max(x), n)
        k = 3 if len(x) > 3 else max(1, len(x) - 1)
        ys = make_interp_spline(x, y, k=k)(xs)
        return xs, ys
    except Exception:
        return np.array(x), np.array(y)


# ========================= Fig 1: beta two-arm collapse =========================
def fig_beta():
    # Both arms aligned to step150. beta=0.1 = ours uniform (FC1); beta=0 = standard (beta0), provisional.
    # step120 (from beta0 local-step30 ckpt) to be added once evaluated.
    b0_s, b0_a = [30, 60, 90, 120, 150], [66.22, 65.67, 63.19, 60.84, 55.61]   # beta=0 collapses (5-pt, 2026-07)
    b1_s, b1_a = [30, 60, 90, 120, 150], [66.22, 65.92, 67.04, 65.91, 65.38]   # beta=0.1 ours (FC1) stays flat
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(b0_s, b0_a, "o-", color=C_ALT, lw=2.2, ms=7, label=r"$\beta=0$ (no support)")
    ax.plot(b1_s, b1_a, "s-", color=C_OURS, lw=2.2, ms=7, label=r"$\beta=0.1$ (ours)")
    for x, y in zip(b0_s, b0_a): ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=10, color=C_ALT)
    for x, y in zip(b1_s, b1_a): ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10, color=C_OURS)
    ax.set_xlabel("Training step"); ax.set_ylabel("Seven-benchmark Acc")
    ax.set_xticks([30, 60, 90, 120, 150]); ax.set_ylim(53, 69)
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/beta_collapse.pdf"); plt.close(fig)
    print("wrote beta_collapse.pdf")


# ========================= Fig 2: alpha curve =========================
def fig_alpha():
    a = [0, 0.5, 1.0, 1.25, 1.5, 2.0]
    acc = [64.31, 63.56, 67.04, 67.04, 66.14, 64.32]  # alpha=1.0 = main-table 67.04; others re-judged
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(a, acc, "o-", color=C_OURS, lw=2.0, ms=8)  # straight segments, no spline
    ax.axhspan(66.0, 67.2, color=C_OURS, alpha=0.06)
    ax.annotate("plateau", (1.30, 67.3), fontsize=11, color=C_OURS)
    for x, y in zip(a, acc): ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10)
    ax.axvline(1.0, color="#888", ls="--", lw=1.0)
    ax.set_xlabel(r"Contrastive strength $\alpha$"); ax.set_ylabel("Seven-benchmark Acc")
    ax.set_xticks(a); ax.set_ylim(60, 68)  # y from 60 so the alpha=0.5 dip shows clearly
    fig.tight_layout(); fig.savefig(f"{OUT}/alpha_curve.pdf"); plt.close(fig)
    print("wrote alpha_curve.pdf")


# ========================= Fig 3: anchor language drift (smoothed) =========================
def fig_drift():
    step = [10, 30, 50, 70, 90]
    no_anchor = [7.8, 12.5, 14.1, 16.4, 26.6]
    anchor    = [7.0, 11.7, 6.2, 8.6, 21.9]
    diff = [n - a for n, a in zip(no_anchor, anchor)]  # drift removed by the anchor
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(step, diff, "o-", color=C_3RD, lw=2.4, ms=7)
    ax.fill_between(step, 0, diff, color=C_3RD, alpha=0.12)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("Training step"); ax.set_ylabel(r"Drift reduced by anchor (\%)")
    ax.set_xticks(step); ax.set_ylim(-1, 10)
    fig.tight_layout(); fig.savefig(f"{OUT}/anchor_drift.pdf"); plt.close(fig)
    print("wrote anchor_drift.pdf")


def fig_combined():
    """3 panels in one row (for a single full-width figure in the paper)."""
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.5))
    # (a) beta
    ax[0].plot([30, 60, 90, 120, 150], [66.22, 65.67, 63.19, 60.84, 55.61], "o-", color=C_ALT, lw=2, ms=6, label=r"$\beta=0$")
    ax[0].plot([30, 60, 90, 120, 150], [66.22, 65.92, 67.04, 65.91, 65.38], "s-", color=C_OURS, lw=2, ms=6, label=r"$\beta=0.1$ (ours)")
    ax[0].set_xlabel("Training step"); ax[0].set_ylabel("Seven-benchmark Acc")
    ax[0].set_title(r"(a) Plausibility support $\beta$"); ax[0].set_xticks([30, 60, 90, 120, 150]); ax[0].legend(fontsize=10)
    # (b) alpha
    a = [0, 0.5, 1.0, 1.25, 1.5, 2.0]; acc = [64.31, 63.56, 67.04, 67.04, 66.14, 64.32]  # alpha=1.0 = main-table 67.04
    ax[1].plot(a, acc, "o-", color=C_OURS, lw=2, ms=7)
    ax[1].axvline(1.0, color="#888", ls="--", lw=1); ax[1].axhspan(66.0, 67.2, color=C_OURS, alpha=0.06)
    ax[1].set_xlabel(r"Contrastive strength $\alpha$"); ax[1].set_ylabel("Seven-benchmark Acc")
    ax[1].set_title(r"(b) Contrastive strength $\alpha$"); ax[1].set_xticks(a); ax[1].set_ylim(60, 68)
    # (c) drift
    step = [10, 30, 50, 70, 90]
    na = [7.8, 12.5, 14.1, 16.4, 26.6]; an = [7.0, 11.7, 6.2, 8.6, 21.9]
    diff = [n - a for n, a in zip(na, an)]
    ax[2].plot(step, diff, "o-", color=C_3RD, lw=2.4, ms=7)
    ax[2].fill_between(step, 0, diff, color=C_3RD, alpha=0.12)
    ax[2].axhline(0, color="#888", lw=0.8)
    ax[2].set_xlabel("Training step"); ax[2].set_ylabel(r"Drift reduced by anchor (\%)")
    ax[2].set_title("(c) Anchor reduces language drift"); ax[2].set_xticks(step); ax[2].set_ylim(-1, 10)
    fig.tight_layout(); fig.savefig(f"{OUT}/combined_ablation.pdf"); plt.close(fig)
    print("wrote combined_ablation.pdf")


def fig_stepcurve():
    """Two side-by-side per-benchmark step curves; each panel plots OPSD (FC4) vs ours (FC1)."""
    step = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    mmstar_ours = [60.93, 63.07, 60.93, 61.87, 61.93, 62.00, 61.93, 63.60, 63.73, 62.47, 62.87, 61.60, 61.40, 62.60, 61.93]
    mmstar_opsd = [61.07, 58.93, 58.27, 59.53, 59.80, 59.47, 59.40, 59.93, 56.87, 60.00, 60.13, 59.73, 59.20, 59.53, 59.67]
    mv_ours = [66.50, 64.90, 64.80, 66.70, 65.60, 65.50, 65.90, 67.00, 66.10, 66.20, 67.10, 65.40, 65.90, 64.40, 63.90]
    mv_opsd = [62.00, 63.60, 62.00, 62.20, 62.30, 63.30, 63.20, 64.50, 62.80, 61.80, 61.50, 61.30, 59.50, 59.10, 59.30]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (yo, yp, title) in zip(ax, [(mmstar_ours, mmstar_opsd, "MMStar"), (mv_ours, mv_opsd, "MathVista")]):
        a.plot(step, yo, "o-", color=C_OURS, lw=2, ms=5, label="ours")
        a.plot(step, yp, "s-", color=C_ALT, lw=2, ms=5, label="OPSD")
        a.set_title(title); a.set_xlabel("Training step"); a.set_ylabel("Accuracy")
        a.set_xticks([10, 50, 90, 130]); a.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/step_curve.pdf"); plt.close(fig)
    print("wrote step_curve.pdf")


def fig_stepcurve_7bench():
    """Seven-benchmark aggregate Acc over training steps: ours (FC1) vs OPSD (FC4).
    Same style as fig_stepcurve so the three curves match visually."""
    import pandas as pd
    base = "/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/docs/fce_curves_20260723"
    o = pd.read_csv(f"{base}/FC1_ours.csv"); b = pd.read_csv(f"{base}/FC4_baseline.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(o.step, o.Acc_7bench, "o-", color=C_OURS, lw=2, ms=5, label="ours")
    ax.plot(b.step, b.Acc_7bench, "s-", color=C_ALT, lw=2, ms=5, label="OPSD")
    ax.set_xlabel("Training step"); ax.set_ylabel("Seven-benchmark Acc")
    ax.set_xticks([10, 50, 90, 130]); ax.set_ylim(56, 69); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/step_curve_7bench.pdf"); plt.close(fig)
    print("wrote step_curve_7bench.pdf")


def fig_stepcurve_row():
    """3-panel row (7-benchmark | MMStar | MathVista), all read from the fce CSVs.
    This is the figure actually used in the paper (step_curve_row.pdf, fig:step-curve).
    MMStar panel: legend lower-left + lower ymin to avoid overlap."""
    import pandas as pd
    b = "/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/docs/fce_curves_20260723"
    o = pd.read_csv(f"{b}/FC1_ours.csv"); c = pd.read_csv(f"{b}/FC4_baseline.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    def d(a, title, col, loc="best", ymin=None):
        a.plot(o.step, o[col], "o-", color=C_OURS, lw=2, ms=5, label="ours")
        a.plot(c.step, c[col], "s-", color=C_ALT, lw=2, ms=5, label="OPSD")
        a.set_title(title); a.set_xlabel("Training step"); a.set_ylabel("Accuracy"); a.set_xticks([10, 50, 90, 130])
        if ymin is not None: a.set_ylim(bottom=ymin)
        a.legend(loc=loc)
    d(ax[0], "Seven-benchmark", "Acc_7bench")
    d(ax[1], "MMStar", "MMStar", loc="lower left", ymin=55)
    d(ax[2], "MathVista", "MathVista")
    fig.tight_layout(); fig.savefig(f"{OUT}/step_curve_row.pdf"); plt.close(fig)
    print("wrote step_curve_row.pdf")


if __name__ == "__main__":
    fig_beta(); fig_alpha(); fig_drift(); fig_combined(); fig_stepcurve(); fig_stepcurve_7bench(); fig_stepcurve_row()
    print("all figures ->", OUT)
