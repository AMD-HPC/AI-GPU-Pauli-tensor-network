#!/usr/bin/env python3
"""
make_gif.py — animate n(r,t) Pauli weight spreading from CppSim --measure=all logs.

Usage:
  # Single grid — one GIF:
  python make_gif.py 4x4_chi020.log -o spreading_4x4.gif

  # Multiple grids side-by-side — comparison GIF:
  python make_gif.py 3x3_chi020.log 4x4_chi020.log 5x5_chi020.log -o comparison.gif

Options:
  -o / --output   output filename          (default: spreading.gif)
  --fps           frames per second        (default: 3)
  --vmax          colorbar maximum         (default: auto from data)
  --title         figure title             (default: from filenames)
  --dpi           output resolution        (default: 120)
"""

import argparse
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from pathlib import Path


# ── parser ────────────────────────────────────────────────────────────────────

def parse_log(path):
    """
    Parse one CppSim --measure=all log file.

    Returns a list of dicts, one per layer (layer 0 excluded):
      {layer, ct, chi_min, chi_max, norm, n_off, se_over_logx, sweeps, grid}

    `grid` is a 2D numpy array of non-identity Pauli weights.
    Grid size (nrows, ncols) is read from the header line "Grid: NxM".
    """
    text = Path(path).read_text()

    # Grid dimensions from header
    m = re.search(r"Grid:\s+(\d+)x(\d+)", text)
    if not m:
        sys.exit(f"[{path}] cannot find 'Grid: NxM' in header")
    nrows, ncols = int(m.group(1)), int(m.group(2))

    layers = []

    # Each measured layer has a data row followed by an "n(r,t)" block.
    # Row pattern (tolerant of variable whitespace):
    row_pat = re.compile(
        r"^\s*(\d+)\s+"          # layer
        r"(\d+)\s+"              # chi_min
        r"(\d+)\s+"              # chi_max
        r"([0-9.eE+\-]+)\s+"    # C(t)
        r"[0-9.eE+\-]+\s+"      # total(s)
        r"[0-9.eE+\-]+\s+"      # trotter(ms)
        r"[0-9.eE+\-]+\s+"      # bp(ms)
        r"[0-9.eE+\-]+\s+"      # Cfn(ms)
        r"[0-9]+\s+"             # sweeps
        r"[0-9.eE+\-]+\s+"      # delta
        r"[0-9.eE+\-]+\s+"      # gate_err
        r"([0-9.eE+\-]+)\s+"    # Se/logX
        r"[0-9.eE+\-]+\s+"      # gap
        r"([0-9.eE+\-]+|inf)\s+"    # norm (may be inf)
        r"([0-9.eE+\-]+)"           # n_off
        r".*$",                  # ignore any extra columns (OSEE/Smax etc)
        re.MULTILINE,
    )

    grid_pat = re.compile(
        r"n\(r,t\) \[non-identity Pauli weight\]:(.*?)(?=\n\s*\[Trotter\]|\n\s*\n|\Z)",
        re.DOTALL,
    )

    for rm in row_pat.finditer(text):
        layer     = int(rm.group(1))
        if layer == 0:
            continue
        chi_min   = int(rm.group(2))
        chi_max   = int(rm.group(3))
        ct        = float(rm.group(4))
        se        = float(rm.group(5))
        norm      = float(rm.group(6))
        n_off     = float(rm.group(7))

        # Find the n(r,t) grid that follows this row in the text
        pos = rm.end()
        gm  = grid_pat.search(text, pos)
        if gm is None:
            continue  # layer with no spatial grid (shouldn't happen with --measure=all)

        numbers = re.findall(r"[0-9]+\.[0-9]+", gm.group(1))
        if len(numbers) != nrows * ncols:
            continue  # malformed grid

        grid = np.array([float(x) for x in numbers]).reshape(nrows, ncols)

        layers.append(dict(
            layer=layer, chi_min=chi_min, chi_max=chi_max,
            ct=ct, se=se, norm=norm, n_off=n_off, grid=grid,
        ))

    if not layers:
        sys.exit(f"[{path}] no spatial layers found — run with --measure=spatial or --measure=all")

    return nrows, ncols, layers


# ── plotting ──────────────────────────────────────────────────────────────────

def make_gif(log_files, output, fps=3, vmax=None, suptitle=None, dpi=120):
    # Parse all files
    datasets = []
    for f in log_files:
        nrows, ncols, layers = parse_log(f)
        label = Path(f).stem
        datasets.append((label, nrows, ncols, layers))

    n_panels = len(datasets)
    n_frames  = min(len(d[3]) for d in datasets)

    # Per-panel vmax so each grid uses its own color scale
    vmaxes = []
    for d in datasets:
        v = max(d[3][t]["grid"].max() for t in range(n_frames))
        vmaxes.append(max(v, 1e-6))
    if vmax is not None:
        vmaxes = [vmax] * len(datasets)  # override with user value

    fig_w = 4.0 * n_panels
    fig_h = 4.0
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), dpi=dpi)
    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor("#1a1a2e")

    vmin_log = 1e-5   # log scale floor — anything below this is black

    ims   = []
    texts = []
    for ax, (label, nrows, ncols, _), vm in zip(axes, datasets, vmaxes):
        ax.set_facecolor("#1a1a2e")
        norm = mcolors.LogNorm(vmin=vmin_log, vmax=max(vm, vmin_log*10))
        im = ax.imshow(
            np.full((nrows, ncols), vmin_log),
            norm=norm,
            cmap="inferno",
            interpolation="nearest",
            origin="upper",
        )
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log n(r,t)", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # Grid lines between cells
        for x in np.arange(-0.5, ncols, 1):
            ax.axvline(x, color="white", linewidth=0.3, alpha=0.4)
        for y in np.arange(-0.5, nrows, 1):
            ax.axhline(y, color="white", linewidth=0.3, alpha=0.4)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=10)

        txt = ax.text(
            0.5, -0.08, "", transform=ax.transAxes,
            ha="center", va="top", color="#e0e0e0", fontsize=8,
        )
        ims.append(im)
        texts.append(txt)

    if suptitle:
        fig.suptitle(suptitle, color="white", fontsize=11, y=1.02)

    plt.tight_layout()

    def update(frame_idx):
        artists = []
        for im, txt, (label, nrows, ncols, layers) in zip(ims, texts, datasets):
            if frame_idx >= len(layers):
                continue
            rec = layers[frame_idx]
            im.set_data(np.clip(rec["grid"], vmin_log, None))
            txt.set_text(
                f"L={rec['layer']}  C(t)={rec['ct']:.4f}  "
                f"χ={rec['chi_min']}–{rec['chi_max']}\n"
                f"Se/logχ={rec['se']:.3f}  n_off={rec['n_off']:.4f}"
            )
            artists += [im, txt]
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=True
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(output, writer=writer)
    plt.close(fig)
    print(f"Saved {output}  ({n_frames} frames, {fps} fps)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", metavar="LOG", help="CppSim log file(s)")
    ap.add_argument("-o", "--output", default="spreading.gif")
    ap.add_argument("--fps",   type=int,   default=3)
    ap.add_argument("--vmax",  type=float, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi",   type=int,   default=120)
    args = ap.parse_args()

    make_gif(
        log_files=args.logs,
        output=args.output,
        fps=args.fps,
        vmax=args.vmax,
        suptitle=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
