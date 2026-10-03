#!/usr/bin/env python3
"""
compute_dispersion.py — spatial dispersion metrics from CppSim --measure=all logs.

For large grids C(t)→0 (quantum scrambling), so C(t) alone is uninformative.
This script extracts quantitative spreading metrics from the n(r,t) spatial weight
maps written by --measure=all, providing a richer picture of operator dynamics.

Metrics computed per layer:
  R_L1    Mean Manhattan distance from center site (light-cone radius proxy)
  R_L2    RMS Euclidean distance (radius of gyration)
  S_ent   Shannon entropy of n(r,t)/total  (0 = localized, log(N²) = uniform)
  S_frac  S_ent / log(N²)  (0→1 normalized; 1 = fully scrambled)
  weight  Total Pauli weight (should stay ≈1 if norm is healthy)

Usage:
  python3 compute_dispersion.py LOG [LOG ...]         # print table
  python3 compute_dispersion.py LOG --plot -o fig.png # also plot R_L1(t)
  python3 compute_dispersion.py logs/mi355x_20260519/*_all.log --plot
"""

import sys
import re
import argparse
import numpy as np
from pathlib import Path


# ── parser ────────────────────────────────────────────────────────────────────

_NUM = r"(?:[0-9.eE+\-]+|inf|nan)"

def parse_all_log(path):
    """Return list of per-layer dicts with grid and metrics."""
    text = Path(path).read_text(errors="replace")

    m = re.search(r"Grid:\s+(\d+)x(\d+)", text)
    if not m:
        return None, None, None
    nrows, ncols = int(m.group(1)), int(m.group(2))

    m = re.search(r"chi_max=(\d+)", text)
    chi_max = int(m.group(1)) if m else 0

    m = re.search(r"Center site: \((\d+),(\d+)\)", text)
    center = (int(m.group(1)), int(m.group(2))) if m else (nrows // 2, ncols // 2)

    row_pat = re.compile(
        r"^\s*(\d+)\s+"          # layer
        r"\d+\s+"                # chi_min
        r"\d+\s+"                # chi_max
        r"([0-9.eE+\-]+)\s+"    # C(t)
        + _NUM + r"\s+"          # total(s)
        + _NUM + r"\s+"          # trotter(ms)
        + _NUM + r"\s+"          # bp(ms)
        + _NUM + r"\s+"          # Cfn(ms)
        + r"\d+\s+"              # sweeps
        + _NUM + r"\s+"          # delta
        + _NUM + r"\s+"          # gate_err
        + r"([0-9.eE+\-]+)\s+"  # Se/logX
        + _NUM + r"\s+"          # gap
        + _NUM + r"\s+"          # norm
        + _NUM,                  # n_off
        re.MULTILINE,
    )

    grid_pat = re.compile(
        r"n\(r,t\) \[non-identity Pauli weight\]:(.*?)"
        r"(?=\n\s*\[Trotter\]|\n\s*\n|\Z)",
        re.DOTALL,
    )

    ii, jj = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
    dist_l1 = np.abs(ii - center[0]) + np.abs(jj - center[1])
    dist_l2 = np.sqrt((ii - center[0])**2 + (jj - center[1])**2)
    log_n2   = np.log(nrows * ncols)

    layers = []
    for rm in row_pat.finditer(text):
        layer = int(rm.group(1))
        if layer == 0:
            continue
        ct = float(rm.group(2))

        gm = grid_pat.search(text, rm.end())
        if gm is None:
            layers.append(dict(layer=layer, ct=ct,
                               R_L1=None, R_L2=None,
                               S_ent=None, S_frac=None, weight=None))
            continue

        numbers = re.findall(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?",
                              gm.group(1))
        if len(numbers) != nrows * ncols:
            layers.append(dict(layer=layer, ct=ct,
                               R_L1=None, R_L2=None,
                               S_ent=None, S_frac=None, weight=None))
            continue

        grid = np.array([float(x) for x in numbers]).reshape(nrows, ncols)
        total = grid.sum()

        if total < 1e-14:
            layers.append(dict(layer=layer, ct=ct,
                               R_L1=None, R_L2=None,
                               S_ent=None, S_frac=None, weight=float(total)))
            continue

        R_L1 = float(np.sum(grid * dist_l1) / total)
        R_L2 = float(np.sqrt(np.sum(grid * dist_l2**2) / total))

        p = grid / total
        p_nz = p[p > 0]
        S_ent  = float(-np.sum(p_nz * np.log(p_nz)))
        S_frac = S_ent / log_n2

        layers.append(dict(layer=layer, ct=ct,
                           R_L1=R_L1, R_L2=R_L2,
                           S_ent=S_ent, S_frac=S_frac,
                           weight=float(total)))

    meta = dict(nrows=nrows, ncols=ncols, chi_max=chi_max,
                center=center, name=Path(path).stem)
    return meta, layers


# ── table output ──────────────────────────────────────────────────────────────

def print_table(meta, layers):
    name = meta['name']
    N = f"{meta['nrows']}×{meta['ncols']}"
    print(f"\n{'='*72}")
    print(f"  {name}  ({N}  chi={meta['chi_max']}  center={meta['center']})")
    print(f"{'='*72}")
    print(f"{'lay':>4}  {'C(t)':>10}  {'R_L1':>6}  {'R_L2':>6}  "
          f"{'S_frac':>7}  {'weight':>8}")
    print(f"{'----':>4}  {'----------':>10}  {'------':>6}  {'------':>6}  "
          f"{'-------':>7}  {'--------':>8}")
    for row in layers:
        L   = row['layer']
        ct  = row['ct']
        R1  = f"{row['R_L1']:.2f}"  if row['R_L1']  is not None else "  —  "
        R2  = f"{row['R_L2']:.2f}"  if row['R_L2']  is not None else "  —  "
        Sf  = f"{row['S_frac']:.3f}" if row['S_frac'] is not None else "  —   "
        wt  = f"{row['weight']:.4f}" if row['weight'] is not None else "  —    "
        print(f"{L:>4}  {ct:>10.6f}  {R1:>6}  {R2:>6}  {Sf:>7}  {wt:>8}")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_dispersion(all_meta, all_layers, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DARK = "#0f1117"
    TEXT = "#e8e8f0"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor(DARK)
    for ax in axes:
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(all_meta)))

    for (meta, layers), col in zip(zip(all_meta, all_layers), colors):
        t    = np.array([r['layer'] * 0.1 for r in layers])
        R1   = np.array([r['R_L1']  if r['R_L1']  is not None else np.nan
                         for r in layers])
        Sf   = np.array([r['S_frac'] if r['S_frac'] is not None else np.nan
                         for r in layers])
        ct   = np.array([r['ct'] for r in layers])
        label = f"{meta['nrows']}×{meta['ncols']} χ={meta['chi_max']}"

        mask = ~np.isnan(R1)
        axes[0].plot(t[mask], R1[mask], color=col, lw=1.5, label=label)
        axes[1].plot(t[mask], Sf[mask], color=col, lw=1.5, label=label)
        mask2 = ct > 0
        axes[2].semilogy(t[mask2], ct[mask2], color=col, lw=1.5, label=label)

    labels = [r"$\langle r \rangle_{L1}$ (lattice sites)",
              r"Entropy fraction $S / \log N^2$",
              r"$C(t)$"]
    xlabels = [r"$t = n\,\delta t$"] * 3

    for ax, yl, xl in zip(axes, labels, xlabels):
        ax.set_ylabel(yl, color=TEXT, fontsize=9)
        ax.set_xlabel(xl, color=TEXT, fontsize=9)
        ax.grid(True, alpha=0.2, lw=0.5)
        if len(all_meta) > 1:
            ax.legend(fontsize=7, framealpha=0.15,
                      labelcolor=TEXT, loc='best')

    axes[0].set_title("Mean spreading radius",      color=TEXT, fontsize=10)
    axes[1].set_title("Entropy (scrambling degree)", color=TEXT, fontsize=10)
    axes[2].set_title("Autocorrelator C(t)",         color=TEXT, fontsize=10)

    plt.tight_layout(pad=1.5)
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"Saved {output}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", metavar="LOG")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("-o", "--output", default="fig_dispersion.png")
    args = ap.parse_args()

    all_meta, all_layers = [], []
    for path in args.logs:
        meta, layers = parse_all_log(path)
        if meta is None:
            print(f"[skip] {path} — no Grid header found", file=sys.stderr)
            continue
        if not layers:
            print(f"[skip] {path} — no spatial layers parsed", file=sys.stderr)
            continue
        print_table(meta, layers)
        all_meta.append(meta)
        all_layers.append(layers)

    if args.plot and all_meta:
        plot_dispersion(all_meta, all_layers, args.output)


if __name__ == "__main__":
    main()
