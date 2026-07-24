#!/usr/bin/env python3
"""Render visual-dependency highlight JSONs (from visual_dependency_highlight.py)
into Fig.5-style highlighted-text figures: 3 rows (Base / OPSD / ours), each token
shaded by delta_t = logp(img) - logp(black). Darker = more visually dependent.
No GPU needed. Outputs PNG (preview) + PDF (paper) per sample.
"""
import json, glob, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

CW = 0.62      # char width in axis units (monospace)
LH = 1.6       # line height
WIDTH = 92     # wrap width in chars


def layout_row(ax, tokens, deltas, norm, cmap, y0, label, avg):
    x, y = 0.0, y0
    ax.text(-1.0, y0 + LH * 0.15, label, ha="right", va="center",
            fontsize=11, fontweight="bold")
    ax.text(-1.0, y0 - LH * 0.75, f"avg $\\Delta$={avg:+.2f}", ha="right",
            va="center", fontsize=8, color="#555")
    for tok, d in zip(tokens, deltas):
        disp = tok.replace("\n", "↵").replace("\t", " ")
        if disp == "":
            disp = " "
        w = max(len(disp), 1) * CW
        if x + w > WIDTH or tok == "\n":
            x = 0.0
            y -= LH
        shade = cmap(norm(max(0.0, d)))
        ax.add_patch(Rectangle((x, y - LH * 0.5), w, LH * 0.92,
                               facecolor=shade, edgecolor="none"))
        # text color: white on dark background
        lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
        tc = "white" if lum < 0.5 else "black"
        ax.text(x + w * 0.5, y, disp, ha="center", va="center",
                fontsize=8.5, family="monospace", color=tc)
        x += w
    return y - LH  # bottom y


def render(data, meta, out_png, out_pdf):
    tokens = data["tokens"]
    D = {t: np.array(data["deltas"][t]) for t in ["base", "opsd", "ours"]}
    n = min(len(tokens), *[len(D[t]) for t in D])
    tokens = tokens[:n]
    D = {t: D[t][:n] for t in D}
    vmax = max(np.percentile(D[t], 95) for t in D)
    vmax = max(vmax, 1e-3)
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("Blues")

    # estimate number of wrapped lines to size the figure
    def nlines(toks):
        x, ln = 0.0, 1
        for tk in toks:
            disp = tk.replace("\n", "↵")
            w = max(len(disp), 1) * CW
            if x + w > WIDTH or tk == "\n":
                x = 0.0; ln += 1
            x += w
        return ln
    lines_each = nlines(tokens)
    total_lines = lines_each * 3 + 4
    fig_h = 0.30 * total_lines + 1.2
    fig, ax = plt.subplots(figsize=(13, fig_h))

    y = 0.0
    gap = LH * 1.4
    for t, lab in [("base", "Base"), ("opsd", "OPSD"), ("ours", r"$\bf{ours}$")]:
        y = layout_row(ax, tokens, D[t], norm, cmap, y, lab, D[t].mean())
        y -= gap

    ax.set_xlim(-9, WIDTH + 1)
    ax.set_ylim(y + LH, LH * 2)
    ax.axis("off")
    title = f"{meta.get('benchmark','')}  |  {data['id']}  |  GT = {meta.get('gt','?')}  |  control = {data.get('control','black')}"
    ax.set_title(title, fontsize=12, loc="left")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(r"$\Delta_t=\log p(\cdot|\mathrm{img})-\log p(\cdot|\mathrm{black})$", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="docs/vdh_out")
    ap.add_argument("--manifest", default="docs/vdh_manifest.jsonl")
    ap.add_argument("--out-dir", default="docs/vdh_fig")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {}
    if os.path.exists(args.manifest):
        for l in open(args.manifest):
            if l.strip():
                m = json.loads(l); meta[m["id"]] = m
    for f in sorted(glob.glob(os.path.join(args.in_dir, "*.json"))):
        data = json.load(open(f))
        sid = data["id"]
        render(data, meta.get(sid, {}),
               os.path.join(args.out_dir, f"{sid}.png"),
               os.path.join(args.out_dir, f"{sid}.pdf"))
        print("rendered", sid)
    print("-> ", args.out_dir)


if __name__ == "__main__":
    main()
