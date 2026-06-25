#!/usr/bin/env python3
"""
make_diff_figure.py — three-row spatial comparison figure.

Row 1 : n(r,t) from LOG1  (e.g. chi=80, with BP trap)   [inferno, log]
Row 2 : n(r,t) from LOG2  (e.g. chi=40, clean)          [inferno, log]
Row 3 : Δn = LOG1 − LOG2                                 [RdBu_r, diverging]
Bottom: C(t) overlay of both runs

Usage:
  python3 make_diff_figure.py LOG1 LOG2 -o fig_diff.png \
      --layers 1 5 10 15 17 18 20 --trap-layer 17
"""

import argparse
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap
from pathlib import Path

DARK   = "#0f1117"
TEXT   = "#e8e8f0"
GRID   = "#2a2a3a"
ACCENT = "#7ec8e3"
TRAP   = "#ff6b6b"
RECOV  = "#90ee90"
ROW1_C = "#ffb347"   # amber  — chi=80 trapped
ROW2_C = "#7ec8e3"   # blue   — chi=40 clean

# Diverging colormap: blue → black (zero) → red
# Zero maps to black so it vanishes into the dark background
CMAP_DIFF = LinearSegmentedColormap.from_list(
    "bkr",
    [(0.0, "#2255cc"),   # negative peak — blue
     (0.4, "#112266"),   # deep blue
     (0.5, "#000000"),   # zero — black
     (0.6, "#661111"),   # deep red
     (1.0, "#cc2222")],  # positive peak — red
)


def parse_log(path):
    text = Path(path).read_text()

    m = re.search(r"Grid:\s+(\d+)x(\d+)", text)
    if not m:
        sys.exit(f"Cannot find 'Grid: NxM' in {path}")
    nrows, ncols = int(m.group(1)), int(m.group(2))

    m = re.search(r"chi_max=(\d+)", text)
    chi_max = int(m.group(1)) if m else "?"

    m = re.search(r"Center site: \((\d+),(\d+)\)", text)
    center = (int(m.group(1)), int(m.group(2))) if m else None

    row_pat = re.compile(
        r"^\s*(\d+)\s+"          # layer
        r"\d+\s+"                # chi_min
        r"\d+\s+"                # chi_max
        r"([0-9.eE+\-]+)",      # C(t)
        re.MULTILINE,
    )
    grid_pat = re.compile(
        r"n\(r,t\) \[non-identity Pauli weight\]:(.*?)(?=\n\s*\[Trotter\]|\n\s*\n|\Z)",
        re.DOTALL,
    )

    layers = {}
    ct_series = [(0, 1.0)]
    for rm in row_pat.finditer(text):
        layer = int(rm.group(1))
        ct    = float(rm.group(2))
        if layer == 0:
            continue
        ct_series.append((layer, ct))
        pos = rm.end()
        gm  = grid_pat.search(text, pos)
        if gm is None:
            continue
        numbers = re.findall(r"[0-9]+\.[0-9]+", gm.group(1))
        if len(numbers) != nrows * ncols:
            continue
        grid = np.array([float(x) for x in numbers]).reshape(nrows, ncols)
        layers[layer] = dict(ct=ct, grid=grid)

    return dict(nrows=nrows, ncols=ncols, chi_max=chi_max,
                center=center, layers=layers, ct_series=ct_series)


def draw_heatmap_row(fig, gs_row, grids_and_meta, norm, center,
                     row_label, label_color, trap_layer=None,
                     ct_color=ACCENT):
    im_last = None
    axes    = []
    for col, (L, grid, ct) in enumerate(grids_and_meta):
        ax = fig.add_subplot(gs_row[col])
        ax.set_facecolor(DARK)
        axes.append(ax)

        im = ax.imshow(grid, norm=norm, cmap="inferno",
                       interpolation="nearest", origin="upper")
        im_last = im

        for x in np.arange(-0.5, grid.shape[1], 1):
            ax.axvline(x, color=GRID, linewidth=0.4)
        for y in np.arange(-0.5, grid.shape[0], 1):
            ax.axhline(y, color=GRID, linewidth=0.4)

        if center:
            ax.plot(center[1], center[0], "+", color=TEXT,
                    markersize=6, markeredgewidth=0.8, alpha=0.6)

        ax.set_xticks([])
        ax.set_yticks([])

        # C(t) label below
        this_ct_color = ct_color
        if trap_layer is not None:
            if L == trap_layer:
                this_ct_color = TRAP
        ax.text(0.5, -0.05, f"$C={ct:.4f}$",
                transform=ax.transAxes, ha="center", va="top",
                color=this_ct_color, fontsize=7.5)

        # "trapped" banner on first row only
        if trap_layer is not None and L == trap_layer and label_color == ROW1_C:
            ax.text(0.5, 0.97, "trapped",
                    transform=ax.transAxes, ha="center", va="top",
                    color=TRAP, fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=DARK,
                              ec=TRAP, alpha=0.85, linewidth=0.8))

    axes[0].set_ylabel(row_label, color=label_color, fontsize=8,
                       labelpad=5, rotation=90, va="center")
    return im_last


def draw_diff_row(fig, gs_row, diffs_and_meta, norm_div, center, t_labels):
    im_last = None
    axes    = []
    for col, (L, diff) in enumerate(diffs_and_meta):
        ax = fig.add_subplot(gs_row[col])
        ax.set_facecolor(DARK)
        axes.append(ax)

        im = ax.imshow(diff, norm=norm_div, cmap=CMAP_DIFF,
                       interpolation="nearest", origin="upper")
        im_last = im

        for x in np.arange(-0.5, diff.shape[1], 1):
            ax.axvline(x, color=GRID, linewidth=0.4)
        for y in np.arange(-0.5, diff.shape[0], 1):
            ax.axhline(y, color=GRID, linewidth=0.4)

        if center:
            ax.plot(center[1], center[0], "+", color=DARK,
                    markersize=6, markeredgewidth=0.8, alpha=0.7)

        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_ylabel(r"$\Delta n$" + "\n(trap − clean)",
                       color=TEXT, fontsize=8, labelpad=5,
                       rotation=90, va="center")
    return im_last


def make_figure(log1_path, log2_path, snap_layers, output,
                trap_layer=None, dpi=150, vmin=1e-4,
                row1_label=None, row2_label=None):

    d1 = parse_log(log1_path)
    d2 = parse_log(log2_path)

    nrows  = d1["nrows"]
    ncols  = d1["ncols"]
    center = d1["center"]

    # Filter to layers present in both
    snap_layers = [L for L in snap_layers
                   if L in d1["layers"] and L in d2["layers"]]
    if not snap_layers:
        sys.exit("No layers found in both logs.")
    n_snaps = len(snap_layers)

    # Gather grids
    gm1 = [(L, d1["layers"][L]["grid"], d1["layers"][L]["ct"])
            for L in snap_layers]
    gm2 = [(L, d2["layers"][L]["grid"], d2["layers"][L]["ct"])
            for L in snap_layers]
    diffs = [(L, d1["layers"][L]["grid"] - d2["layers"][L]["grid"])
             for L in snap_layers]

    # Shared log norm for rows 1 & 2
    all_grids = [g for _, g, _ in gm1] + [g for _, g, _ in gm2]
    vmax_n = max(g.max() for g in all_grids)
    norm_log = LogNorm(vmin=vmin, vmax=vmax_n, clip=True)

    # Shared diverging norm for row 3 — fixed across all 7 panels
    abs_max = max(np.abs(diff).max() for _, diff in diffs)
    abs_max = max(abs_max, 1e-5)
    norm_div = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # ── layout ────────────────────────────────────────────────────────────────
    panel_w = 1.9
    fig_w   = panel_w * n_snaps + 1.2
    fig_h   = 10.5
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(DARK)

    outer = gridspec.GridSpec(4, 1,
                              height_ratios=[1.0, 1.0, 1.0, 0.9],
                              hspace=0.50, figure=fig)

    def make_strip(row_idx):
        return gridspec.GridSpecFromSubplotSpec(
            1, n_snaps, subplot_spec=outer[row_idx], wspace=0.06)

    # t= labels on top of row 1 only — add via title of each ax later
    # Row 1: chi=80 (trapped)
    lbl1 = row1_label if row1_label else f"$\\chi_{{\\max}}={d1['chi_max']}$ (trap)"
    lbl2 = row2_label if row2_label else f"$\\chi_{{\\max}}={d2['chi_max']}$ (clean)"

    gs1 = make_strip(0)
    im1 = draw_heatmap_row(
        fig, gs1, gm1, norm_log, center,
        lbl1, ROW1_C, trap_layer=trap_layer, ct_color=ROW1_C)

    # Add t= labels on row 1
    for col, (L, _, _) in enumerate(gm1):
        ax = fig.axes[col]   # row 1 axes are the first n_snaps added
        ax.set_title(f"$t={L*0.1:.1f}$", color=TEXT, fontsize=8.5, pad=2)

    gs2 = make_strip(1)
    im2 = draw_heatmap_row(
        fig, gs2, gm2, norm_log, center,
        lbl2, ROW2_C, ct_color=ROW2_C)

    # Shared colorbar for rows 1+2
    cbar1_ax = fig.add_axes([0.93, 0.60, 0.011, 0.34])
    cb1 = fig.colorbar(im1, cax=cbar1_ax)
    cb1.set_label(r"$n(\mathbf{r},t)$", color=TEXT, fontsize=8)
    cb1.ax.yaxis.set_tick_params(color=TEXT, labelsize=6.5)
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color=TEXT)

    # Row 3: difference
    gs3 = make_strip(2)
    im3 = draw_diff_row(fig, gs3, diffs, norm_div, center,
                        [f"$t={L*0.1:.1f}$" for L in snap_layers])

    cbar3_ax = fig.add_axes([0.93, 0.32, 0.011, 0.22])
    cb3 = fig.colorbar(im3, cax=cbar3_ax)
    cb3.set_label(r"$\Delta n$", color=TEXT, fontsize=8)
    cb3.ax.yaxis.set_tick_params(color=TEXT, labelsize=6.5)
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color=TEXT)

    # Diff row note
    diff_ax0 = fig.axes[2 * n_snaps]
    short1 = f"$\\chi={d1['chi_max']}$"
    short2 = f"$\\chi={d2['chi_max']}$"
    diff_ax0.text(
        0.0, -0.10,
        f"red = {short1} > {short2}   ·   blue = {short1} < {short2}   "
        f"·   $|\\Delta n|_{{\\max}}={abs_max:.4f}$",
        transform=diff_ax0.transAxes, ha="left", va="top",
        color=TEXT, fontsize=7.5, alpha=0.8)

    # ── C(t) panel ────────────────────────────────────────────────────────────
    ax_ct = fig.add_subplot(outer[3])
    ax_ct.set_facecolor(DARK)

    for d, col, lbl in [(d1, ROW1_C, lbl1),
                         (d2, ROW2_C, lbl2)]:
        t_arr = np.array([l for l, _ in d["ct_series"]]) * 0.1
        c_arr = np.array([c for _, c in d["ct_series"]])
        ax_ct.semilogy(t_arr, c_arr, color=col, linewidth=1.4, label=lbl)

    for L in snap_layers:
        ax_ct.axvline(L * 0.1, color=TEXT, linewidth=0.5,
                      linestyle=":", alpha=0.4)

    if trap_layer is not None:
        t_trap = trap_layer * 0.1
        ct_val = d1["layers"][trap_layer]["ct"]
        ax_ct.plot(t_trap, ct_val, "x", color=TRAP,
                   markersize=9, markeredgewidth=2.0, zorder=5)

    ax_ct.set_xlabel(r"$t = n\,\delta t$", color=TEXT, fontsize=9)
    ax_ct.set_ylabel(r"$C(t)$", color=TEXT, fontsize=9)
    ax_ct.tick_params(colors=TEXT, labelsize=8)
    for sp in ax_ct.spines.values():
        sp.set_edgecolor(GRID)
    t_max = max(max(l for l, _ in d["ct_series"]) for d in [d1, d2]) * 0.1
    ax_ct.set_xlim(0, t_max + 0.05)
    leg = ax_ct.legend(fontsize=8, framealpha=0.2, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TEXT)

    # ── title ─────────────────────────────────────────────────────────────────
    s1 = f"$\\chi={d1['chi_max']}$ (BP trap)" if trap_layer is not None else f"$\\chi={d1['chi_max']}$"
    s2 = f"$\\chi={d2['chi_max']}$ (clean)"   if trap_layer is not None else f"$\\chi={d2['chi_max']}$"
    fig.suptitle(
        rf"$n(\mathbf{{r}},t)$ comparison: {s1} vs {s2}"
        rf"  —  {nrows}$\times${ncols} grid",
        color=TEXT, fontsize=10, y=1.005,
    )

    plt.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Saved {output}  ({n_snaps} columns)")
    print(f"  Δn range across all layers: "
          f"[{min(d.min() for _,d in diffs):.5f}, "
          f"{max(d.max() for _,d in diffs):.5f}]")
    for L, diff in diffs:
        print(f"  L{L:2d}  max|Δn|={np.abs(diff).max():.5f}"
              + ("  ← TRAP" if L == trap_layer else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log1", help="First log  (e.g. chi=80 with trap)")
    ap.add_argument("log2", help="Second log (e.g. chi=40 clean)")
    ap.add_argument("-o", "--output", default="fig_diff.png")
    ap.add_argument("--layers", nargs="+", type=int, required=True,
                    help="Trotter layer indices to show")
    ap.add_argument("--trap-layer", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--vmin", type=float, default=1e-4)
    ap.add_argument("--row1-label", type=str, default=None,
                    help="Y-axis label for row 1 (overrides auto 'chi=N (trap)')")
    ap.add_argument("--row2-label", type=str, default=None,
                    help="Y-axis label for row 2 (overrides auto 'chi=N (clean)')")
    args = ap.parse_args()

    make_figure(
        log1_path=args.log1,
        log2_path=args.log2,
        snap_layers=args.layers,
        output=args.output,
        trap_layer=args.trap_layer,
        dpi=args.dpi,
        vmin=args.vmin,
        row1_label=args.row1_label,
        row2_label=args.row2_label,
    )


if __name__ == "__main__":
    main()
