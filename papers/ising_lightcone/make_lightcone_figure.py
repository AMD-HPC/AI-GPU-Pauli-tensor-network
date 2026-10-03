#!/usr/bin/env python3
"""
make_lightcone_figure.py — static multi-panel lightcone figure for the paper.

Single-log mode (one strip row):
  python3 make_lightcone_figure.py LOG -o fig_lightcone.png --layers 1 5 10 15 20

Two-log comparison mode:
  python3 make_lightcone_figure.py LOG1 LOG2 -o fig_lightcone.png --layers 1 5 10 15 20

Three-log mode (full paper figure):
  python3 make_lightcone_figure.py LOG1 LOG2 LOG3 -o fig_lightcone.png --layers 1 5 10 15 20
  LOG1 → 4×4 chi=80 with trap   (amber)
  LOG2 → 4×4 chi=40 no trap     (steel blue)
  LOG3 → 9×9 chi=20 symmetric   (green)

Options:
  --layers        Trotter layer indices to show (default: auto 6)
  --vmin          log-scale floor               (default: 1e-4)
  --dpi           output resolution             (default: 150)
  --row-labels    override row labels (one per log)
"""

import argparse
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from pathlib import Path


# ── parser ────────────────────────────────────────────────────────────────────

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

    _num = r"(?:[0-9.eE+\-]+|inf|nan)"   # matches finite, inf, nan
    row_pat = re.compile(
        r"^\s*(\d+)\s+"          # layer
        r"\d+\s+"                # chi_min
        r"\d+\s+"                # chi_max
        r"([0-9.eE+\-]+)\s+"    # C(t)
        + _num + r"\s+"          # total(s)
        + _num + r"\s+"          # trotter(ms)
        + _num + r"\s+"          # bp(ms)
        + _num + r"\s+"          # Cfn(ms)
        + r"\d+\s+"              # sweeps
        + _num + r"\s+"          # delta
        + _num + r"\s+"          # gate_err
        + r"([0-9.eE+\-]+)\s+"  # Se/logX
        + _num + r"\s+"          # gap
        + _num + r"\s+"          # norm  ← can be 'inf'
        + _num,                  # n_off
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
        if layer == 0:
            continue
        ct = float(rm.group(2))
        ct_series.append((layer, ct))

        pos = rm.end()
        gm  = grid_pat.search(text, pos)
        if gm is None:
            continue
        numbers = re.findall(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?", gm.group(1))
        if len(numbers) != nrows * ncols:
            continue
        grid = np.array([float(x) for x in numbers]).reshape(nrows, ncols)
        layers[layer] = dict(ct=ct, grid=grid)

    if not layers:
        sys.exit(f"[{path}] no spatial layers found")

    return dict(
        nrows=nrows, ncols=ncols, chi_max=chi_max, center=center,
        layers=layers, ct_series=ct_series,
        label=f"{nrows}×{ncols}  $\\chi_{{\\max}}={chi_max}$",
    )


# ── figure ────────────────────────────────────────────────────────────────────

DARK   = "#0f1117"
TEXT   = "#e8e8f0"
GRID   = "#2a2a3a"
ACCENT = "#7ec8e3"
TRAP   = "#ff6b6b"

# Row label colours: trapped / clean-asymmetric / symmetric
ROW_COLORS = ["#ffb347", "#7ec8e3", "#90ee90"]   # amber / steel-blue / green


def auto_layers(layers_dict, n=6):
    available = sorted(layers_dict.keys())
    idx = np.linspace(0, len(available) - 1, n, dtype=int)
    return [available[i] for i in idx]


def draw_strip(fig, gs_row, dataset, snap_layers, norm, row_label, label_color,
               trap_layer=None):
    """Draw one horizontal strip of heatmap snapshots.

    trap_layer: if set, marks that layer as 'trapped' and the next
                snapshot as 'recovered' with coloured corner banners.
    """
    nrows  = dataset["nrows"]
    ncols  = dataset["ncols"]
    layers = dataset["layers"]
    center = dataset["center"]

    # Determine recovered layer (first snap after trap_layer)
    recovered_layer = None
    if trap_layer is not None:
        later = [L for L in snap_layers if L > trap_layer]
        if later:
            recovered_layer = later[0]

    axes = []
    im   = None
    for col, L in enumerate(snap_layers):
        ax = fig.add_subplot(gs_row[col])
        ax.set_facecolor(DARK)
        axes.append(ax)

        if L not in layers:
            ax.set_visible(False)
            continue

        data = layers[L]["grid"]
        im = ax.imshow(data, norm=norm, cmap="inferno",
                       interpolation="nearest", origin="upper")

        # Grid lines
        for x in np.arange(-0.5, ncols, 1):
            ax.axvline(x, color=GRID, linewidth=0.4)
        for y in np.arange(-0.5, nrows, 1):
            ax.axhline(y, color=GRID, linewidth=0.4)

        # Mark center site
        if center is not None:
            ax.plot(center[1], center[0], "+", color=TEXT,
                    markersize=6, markeredgewidth=0.8, alpha=0.6)

        ax.set_xticks([])
        ax.set_yticks([])

        # t label
        t_val = L * 0.1
        ax.set_title(f"$t={t_val:.1f}$", color=TEXT, fontsize=8.5, pad=2)

        # C(t) below each panel
        ct_color = ACCENT
        if L == trap_layer:
            ct_color = TRAP
        elif L == recovered_layer:
            ct_color = "#90ee90"
        ax.text(0.5, -0.05, f"$C={layers[L]['ct']:.4f}$",
                transform=ax.transAxes, ha="center", va="top",
                color=ct_color, fontsize=7.5)

        # Corner banner: "trapped" / "recovered"
        if L == trap_layer:
            ax.text(0.5, 0.97, "trapped",
                    transform=ax.transAxes, ha="center", va="top",
                    color=TRAP, fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=DARK,
                              ec=TRAP, alpha=0.85, linewidth=0.8))
        elif L == recovered_layer:
            ax.text(0.5, 0.97, "recovered",
                    transform=ax.transAxes, ha="center", va="top",
                    color="#90ee90", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=DARK,
                              ec="#90ee90", alpha=0.85, linewidth=0.8))

    # Row label on left spine of first panel
    axes[0].set_ylabel(row_label, color=label_color, fontsize=8.5,
                       labelpad=6, rotation=90, va="center")

    return im   # return last im for colorbar


def make_figure(log_paths, output, snap_layers=None, dpi=150, vmin=1e-4,
                row_labels=None, trap_layer=None, tstar=None):

    datasets = [parse_log(p) for p in log_paths]
    n_rows   = len(datasets)

    # Resolve snapshot layers from first dataset if not specified
    if snap_layers is None:
        snap_layers = auto_layers(datasets[0]["layers"], n=6)
    else:
        # keep only layers present in ALL datasets
        available = set.intersection(*(set(d["layers"].keys()) for d in datasets))
        snap_layers = [L for L in snap_layers if L in available]
        if not snap_layers:
            # fall back: use layers from first dataset
            snap_layers = auto_layers(datasets[0]["layers"], n=6)

    n_snaps = len(snap_layers)

    # Global vmax
    vmax = max(
        datasets[r]["layers"][L]["grid"].max()
        for r in range(n_rows)
        for L in snap_layers
        if L in datasets[r]["layers"]
    )
    vmax = max(vmax, vmin * 10)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    # Row labels
    if row_labels is None:
        if n_rows == 3:
            row_labels = [
                f"4×4, trapped layer  ({datasets[0]['label']})",
                f"4×4, no trap        ({datasets[1]['label']})",
                f"9×9, C4 symmetric   ({datasets[2]['label']})",
            ]
        elif n_rows == 2:
            row_labels = [
                f"Asymmetric  ({datasets[0]['label']})",
                f"Symmetric   ({datasets[1]['label']})",
            ]
        else:
            row_labels = [d["label"] for d in datasets]
    label_colors = [ROW_COLORS[i % len(ROW_COLORS)] for i in range(n_rows)]

    # ── layout ────────────────────────────────────────────────────────────────
    # Panel height scales with grid aspect so cells stay square
    cell_px   = 2.1                          # width per snapshot panel (inches)
    panel_w   = cell_px * n_snaps + 0.6     # total figure width
    nrows_max = max(d["nrows"] for d in datasets)
    ncols_max = max(d["ncols"] for d in datasets)
    panel_h   = cell_px * nrows_max / ncols_max * n_snaps / n_snaps  # per row
    panel_h   = max(panel_h, 2.8)           # minimum panel height
    fig_h     = panel_h * n_rows + 2.2      # + C(t) subplot
    fig       = plt.figure(figsize=(panel_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(DARK)

    # Outer grid: n_rows strips + 1 C(t) panel
    outer = gridspec.GridSpec(
        n_rows + 1, 1,
        height_ratios=[3.0] * n_rows + [2.0],
        hspace=0.55,
        figure=fig,
    )

    last_im = None
    for row_idx, (dataset, rl, lc) in enumerate(
            zip(datasets, row_labels, label_colors)):
        gs_row = gridspec.GridSpecFromSubplotSpec(
            1, n_snaps,
            subplot_spec=outer[row_idx],
            wspace=0.06,
        )
        tl = trap_layer if row_idx == 0 else None
        im = draw_strip(fig, gs_row, dataset, snap_layers, norm, rl, lc,
                        trap_layer=tl)
        last_im = im

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.38, 0.012, 0.55])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label(r"$n(\mathbf{r},t)$", color=TEXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

    # ── C(t) panel ────────────────────────────────────────────────────────────
    ax_ct = fig.add_subplot(outer[n_rows])
    ax_ct.set_facecolor(DARK)

    for d_idx, (dataset, lc) in enumerate(zip(datasets, label_colors)):
        t_arr = np.array([l for l, _ in dataset["ct_series"]]) * 0.1
        c_arr = np.array([c for _, c in dataset["ct_series"]])
        col   = lc
        lbl   = row_labels[d_idx]
        ax_ct.semilogy(t_arr, c_arr, color=col, linewidth=1.3,
                       label=lbl, zorder=2)

    # Vertical markers at snapshot layers (from first dataset)
    for L in snap_layers:
        t_snap = L * 0.1
        ax_ct.axvline(t_snap, color=TEXT, linewidth=0.5,
                      linestyle=":", alpha=0.45, zorder=1)

    ax_ct.set_xlabel(r"$t = n\,\delta t$", color=TEXT, fontsize=9)
    ax_ct.set_ylabel(r"$C(t)$",            color=TEXT, fontsize=9)
    ax_ct.tick_params(colors=TEXT, labelsize=8)
    for sp in ax_ct.spines.values():
        sp.set_edgecolor(GRID)

    t_max = max(
        max(l for l, _ in d["ct_series"]) * 0.1
        for d in datasets
    )
    ax_ct.set_xlim(0, t_max + 0.05)

    # n* vertical marker
    if tstar is not None and tstar <= t_max:
        ax_ct.axvline(tstar, color=TRAP, linewidth=1.1,
                      linestyle="--", alpha=0.85, zorder=3)
        ax_ct.text(tstar + 0.03 * t_max, ax_ct.get_ylim()[0] * 3,
                   r"$n^*$", color=TRAP, fontsize=8.5, va="bottom")

    # Trapped layer marker on C(t) curve
    if trap_layer is not None:
        t_trap = trap_layer * 0.1
        for d_idx, dataset in enumerate(datasets):
            if t_trap * 10 in dataset["layers"]:
                ct_trap = dataset["layers"][trap_layer]["ct"]
                ax_ct.plot(t_trap, ct_trap, "x", color=TRAP,
                           markersize=8, markeredgewidth=1.8, zorder=5)

    if n_rows > 1:
        leg = ax_ct.legend(fontsize=7.5, framealpha=0.2,
                           labelcolor="linecolor", loc="upper right")
        for text in leg.get_texts():
            text.set_color(TEXT)

    # ── title ─────────────────────────────────────────────────────────────────
    if n_rows == 3:
        title = (r"Operator spreading $n(\mathbf{r},t)$: "
                 r"trapped layer, clean, and C4-symmetric")
    elif n_rows == 2:
        title = (r"Operator spreading $n(\mathbf{r},t)$: "
                 r"asymmetric vs.\ symmetric")
    else:
        title = (rf"Operator spreading $n(\mathbf{{r}},t)$ — "
                 rf"{datasets[0]['nrows']}$\times${datasets[0]['ncols']}, "
                 rf"$\chi_\mathrm{{max}}={datasets[0]['chi_max']}$")

    fig.suptitle(title, color=TEXT, fontsize=10, y=1.005)

    plt.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Saved {output}  ({n_snaps} snapshots, {n_rows} row(s))")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", metavar="LOG",
                    help="1–3 CppSim log files")
    ap.add_argument("-o", "--output", default="fig_lightcone.png")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Trotter layers to show (default: auto 6)")
    ap.add_argument("--vmin",  type=float, default=1e-4)
    ap.add_argument("--dpi",   type=int,   default=150)
    ap.add_argument("--row-labels", nargs="+", default=None,
                    help="Override row labels (one per log file)")
    ap.add_argument("--trap-layer", type=int, default=None,
                    help="Trotter layer index of the BP trap; adds 'trapped'/'recovered' banners")
    ap.add_argument("--tstar", type=float, default=None,
                    help="Lightcone saturation time t* (physical time); adds dashed marker on C(t)")
    args = ap.parse_args()

    if len(args.logs) > 3:
        sys.exit("Provide 1, 2, or 3 log files.")

    make_figure(
        log_paths=args.logs,
        output=args.output,
        snap_layers=args.layers,
        dpi=args.dpi,
        vmin=args.vmin,
        row_labels=args.row_labels,
        trap_layer=args.trap_layer,
        tstar=args.tstar,
    )


if __name__ == "__main__":
    main()
