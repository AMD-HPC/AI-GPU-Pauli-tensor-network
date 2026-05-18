#!/usr/bin/env python3
"""
plot_results.py — CppSim figure generator

Produces three publication-quality figures from CppSim log files:
  Fig 1. C(t) vs time — one curve per (GPU-label, chi) pair, with exp(-αt) fit
          and optional Julia reference overlay.
  Fig 2. Achieved TFLOPS per layer vs layer index — roofline comparison across GPUs.
  Fig 3. Grid scaling — trotter time vs number of bonds (one curve per GPU label).

Usage
-----
  # C(t) and roofline for chi-convergence study:
  python plot_results.py \\
      -MI100 results/8abc3c8695f5_4x4_chi050.log \\
      -MI100 results/8abc3c8695f5_4x4_chi070.log \\
      -MI100 results/8abc3c8695f5_4x4_chi080.log \\
      -MI100 results/8abc3c8695f5_4x4_chi100.log \\
      --julia results/julia_ct_chi50.csv

  # Roofline comparison MI100 vs MI60 at chi=100:
  python plot_results.py \\
      -MI100 results/8abc3c8695f5_4x4_chi100.log \\
      -MI60  results/fastmmw_4x4_chi100.log

  # Grid scaling:
  python plot_results.py \\
      -MI100 results/8abc3c8695f5_3x3_chi050.log \\
      -MI100 results/8abc3c8695f5_4x4_chi050.log \\
      -MI100 results/8abc3c8695f5_5x5_chi050.log \\
      -MI100 results/8abc3c8695f5_6x6_chi050.log \\
      -MI100 results/8abc3c8695f5_7x7_chi050.log \\
      -MI100 results/8abc3c8695f5_8x8_chi050.log \\
      -MI60  results/fastmmw_3x3_chi050.log \\
      -MI60  results/fastmmw_4x4_chi050.log \\
      -MI60  results/fastmmw_5x5_chi050.log \\
      -MI60  results/fastmmw_6x6_chi050.log \\
      -MI60  results/fastmmw_7x7_chi050.log \\
      -MI60  results/fastmmw_8x8_chi050.log

Flags
-----
  -LABEL path   Attach GPU label LABEL to the log file at path. Repeatable.
                LABEL can be any string: MI100, MI60, A100, etc.
  --julia path  Two-column CSV (no header) with columns: t, C_t.
                Plotted as "Julia χ=50" reference in the C(t) figure.
  --out prefix  Output file prefix (default: "fig"). Produces
                {prefix}_ct.pdf, {prefix}_roofline.pdf, {prefix}_grid.pdf.
  --dpi N       Figure DPI (default: 300).
  --show        Display figures interactively in addition to saving.
"""

import sys
import re
import os
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Hardware roofline ceilings ────────────────────────────────────────────────
# Peak FP32 TFLOPS per GPU label prefix (case-insensitive prefix match).
ROOFLINE = {
    "MI100": 184.6,
    "MI60":   29.5,
    "A100":  312.0,
    "H100":  756.0,
}

def roofline_for(label):
    for key, val in ROOFLINE.items():
        if label.upper().startswith(key.upper()):
            return val
    return None

# ── FLOP model (from estimate_memory.py) ─────────────────────────────────────
D = 4

def flops_per_gate(chi):
    """
    FLOPs per apply_binary_gate call (one bond = two sites).

    Cholesky-QR per site has TWO GEMMs:
      (1) Gram matrix:    cv^T × cv   (n×m)×(m×n) → n×n : 2*m*n^2
      (2) Q reconstruct:  cv  × R^-1  (m×n)×(n×n) → m×n : 2*m*n^2
    Total QR per site: 4*m*n^2   (×2 sites = 8*m*n^2)

    POTRF (Cholesky of n×n gram): n^3/3 per site — included but minor.
    SVD of RR (n×n bidiagonal):   10*n^3
    Reconstruct GEMM (×2 sites):  2 * 2*m*n*chi
    """
    m = chi**3
    n = D * chi
    qr    = 4 * m * n**2        # two GEMMs per site (Gram + Q)
    potrf = n**3 // 3           # Cholesky per site
    svd   = 10 * n**3
    gemm  = 2 * 2 * m * n * chi # reconstruction, two sites
    return 2 * (qr + potrf) + svd + gemm

def n_bonds(k, l):
    return k * (l - 1) + (k - 1) * l

# ── Log parser ────────────────────────────────────────────────────────────────

def parse_log(path):
    """
    Returns dict with keys:
      grid_k, grid_l, chi_max, steps, dt, h, J
      layers: list of dicts, one per layer 0..N with keys:
        layer, chi_min, chi_max_live, ct, total_s, trotter_ms, bp_ms, cfn_ms
    """
    result = {
        "path": path,
        "grid_k": None, "grid_l": None,
        "chi_max": None, "steps": None, "dt": None,
        "layers": [],
    }
    header_re = re.compile(
        r"Grid:\s*(\d+)x(\d+)\s+chi_max=(\d+)\s+D=\d+")
    steps_re  = re.compile(
        r"Steps:\s*(\d+)\s+dt=([\d.]+)")
    row_re    = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+"
        r"([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+|-)")

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = header_re.search(line)
            if m:
                result["grid_k"]  = int(m.group(1))
                result["grid_l"]  = int(m.group(2))
                result["chi_max"] = int(m.group(3))
                continue
            m = steps_re.search(line)
            if m:
                result["steps"] = int(m.group(1))
                result["dt"]    = float(m.group(2))
                continue
            m = row_re.match(line)
            if m:
                def fval(s):
                    return float(s) if s != "-" else None
                result["layers"].append({
                    "layer":        int(m.group(1)),
                    "chi_min":      int(m.group(2)),
                    "chi_max_live": int(m.group(3)),
                    "ct":           float(m.group(4)),
                    "total_s":      fval(m.group(5)),
                    "trotter_ms":   fval(m.group(6)),
                    "bp_ms":        fval(m.group(7)),
                    "cfn_ms":       fval(m.group(8)),
                })

    return result

# ── CLI parsing ───────────────────────────────────────────────────────────────

def parse_args(argv):
    """
    Custom parser: collects (-LABEL path) pairs and standard flags.
    Returns namespace with .runs = [(label, path), ...], .julia, .out, .dpi, .show.
    """
    runs   = []
    julia  = None
    out    = "fig"
    dpi    = 300
    show   = False

    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("-") and not a.startswith("--") and len(a) > 1:
            label = a[1:]  # strip leading '-'
            i += 1
            if i >= len(argv):
                sys.exit(f"Error: -{label} requires a file path argument")
            runs.append((label, argv[i]))
        elif a == "--julia":
            i += 1
            julia = argv[i]
        elif a == "--out":
            i += 1
            out = argv[i]
        elif a == "--dpi":
            i += 1
            dpi = int(argv[i])
        elif a == "--show":
            show = True
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            sys.exit(f"Unknown argument: {a}")
        i += 1

    return runs, julia, out, dpi, show

# ── Color / marker assignment ─────────────────────────────────────────────────

LABEL_COLORS = {
    "MI100": "#e84040",  # red
    "MI60":  "#4070c8",  # blue
    "A100":  "#40a860",  # green
    "H100":  "#c87830",  # orange
}
CHI_MARKERS = {50: "o", 70: "s", 80: "^", 100: "D", 120: "P", "other": "x"}

def label_color(label):
    for key, col in LABEL_COLORS.items():
        if label.upper().startswith(key.upper()):
            return col
    # fallback: cycle
    cols = list(LABEL_COLORS.values())
    return cols[hash(label) % len(cols)]

def chi_marker(chi):
    return CHI_MARKERS.get(chi, CHI_MARKERS["other"])

# ── Style ─────────────────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "figure.dpi":        100,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — C(t) plot
# ══════════════════════════════════════════════════════════════════════════════

def load_julia(path):
    """
    Load Julia log.  Handles two formats:
      1. Two-column CSV (no header): t, C_t
      2. Julia simulator table format:
            L    χ   dt(s)  ‖O(t)‖  Re[Tr(O)]  Im[Tr(O)]  Re[C(t)]  Im[C(t)]  gate_err
         detected by the presence of 'Re[C(t)]' in the header line.
    Returns list of (t, C_t) where t = L * dt (dt=0.1 assumed from CppSim convention).
    """
    data = []
    julia_table = False
    dt = 0.1   # default time step

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("─") or line.startswith("!"):
                continue
            if "Re[C(t)]" in line:
                julia_table = True
                continue
            if julia_table:
                # row: L  chi  dt_s  norm  reTr  imTr  reCt  imCt  gate_err
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        L    = int(parts[0])
                        re_ct = float(parts[6])
                        t = L * dt
                        data.append((t, re_ct))
                    except ValueError:
                        pass
            else:
                # CSV fallback
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        data.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return data


def make_ct_figure(runs_data, julia_data, julia_label, out_prefix, dpi, show):
    """
    runs_data: list of (label, parsed_log)
    julia_data: list of (t, ct) or None
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # group runs: if multiple chi for same label, show all
    # find the highest-chi run for the exp fit (most converged)
    best_run = None
    best_chi = 0
    for label, log in runs_data:
        chi = log["chi_max"]
        layers = log["layers"]
        if not layers:
            continue
        ts  = [lay["layer"] * log.get("dt", 0.1) for lay in layers]
        cts = [lay["ct"] for lay in layers]
        color  = label_color(label)
        marker = chi_marker(chi)
        lbl = f"{label} χ={chi}"
        ax.plot(ts, cts, marker=marker, color=color, linewidth=1.4,
                markersize=4, label=lbl, zorder=3)
        if chi > best_chi:
            best_chi = chi
            best_run = (ts, cts)

    # Julia reference
    if julia_data:
        jt  = [r[0] for r in julia_data]
        jct = [r[1] for r in julia_data]
        ax.plot(jt, jct, color="black", linewidth=1.8, linestyle="--",
                marker="*", markersize=6, label=julia_label, zorder=4)


    ax.set_xlabel("Time  $t$")
    ax.set_ylabel("Spin autocorrelation  $C(t)$")
    ax.set_title("Spin autocorrelation $C(t)$ — χ-convergence and GPU comparison")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_yscale("log")
    fig.tight_layout()

    path = f"{out_prefix}_ct.png"
    fig.savefig(path, dpi=dpi)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Roofline / achieved TFLOPS per layer
# ══════════════════════════════════════════════════════════════════════════════

def make_roofline_figure(runs_data, out_prefix, dpi, show):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, log in runs_data:
        k, l   = log["grid_k"], log["grid_l"]
        chi    = log["chi_max"]
        layers = log["layers"]
        if not layers or k is None:
            continue

        peak = roofline_for(label)
        if peak is None:
            print(f"Warning: no peak TFLOPS known for label '{label}', skipping roofline.")
            continue

        nb = n_bonds(k, l)

        xs, ys = [], []
        for lay in layers:
            if lay["layer"] == 0 or lay["trotter_ms"] is None:
                continue
            t_s = lay["trotter_ms"] / 1e3
            if t_s <= 0:
                continue
            chi_live = lay["chi_max_live"]          # actual chi at this layer
            tflops_theoretical = flops_per_gate(chi_live) * nb / 1e12
            achieved_tflops    = tflops_theoretical / t_s
            efficiency         = achieved_tflops / peak  # 0–1
            xs.append(lay["layer"])
            ys.append(efficiency * 100)              # display as percent

        color  = label_color(label)
        marker = chi_marker(chi)
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.3,
                markersize=4, label=f"{label} χ={chi}", zorder=3, alpha=0.85)

    ax.set_xlabel("Trotter layer")
    ax.set_ylabel("FP32 efficiency  (% of peak)")
    ax.set_title("GPU compute efficiency per Trotter layer\n"
                 "(achieved TFLOPS / hardware peak TFLOPS)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:.2g}%"))
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.set_xlim(left=0)
    fig.tight_layout()

    path = f"{out_prefix}_roofline.png"
    fig.savefig(path, dpi=dpi)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Grid scaling (trotter time at saturation vs bonds)
# ══════════════════════════════════════════════════════════════════════════════

def saturation_trotter_ms(log):
    """
    Median trotter_ms of layers where chi_max_live == chi_max (fully saturated),
    excluding None values. Returns None if no saturated layers.
    """
    chi_max = log["chi_max"]
    vals = [
        lay["trotter_ms"]
        for lay in log["layers"]
        if lay["chi_max_live"] == chi_max and lay["trotter_ms"] is not None
    ]
    if not vals:
        return None
    return float(np.median(vals))

def make_grid_figure(runs_data, out_prefix, dpi, show):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # group by label
    by_label = defaultdict(list)
    for label, log in runs_data:
        k, l = log["grid_k"], log["grid_l"]
        if k is None:
            continue
        nb  = n_bonds(k, l)
        sat = saturation_trotter_ms(log)
        if sat is not None:
            by_label[label].append((nb, sat / 1e3))   # seconds

    # sort each group by bond count and plot
    for label, points in sorted(by_label.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = label_color(label)
        ax.plot(xs, ys, marker="o", color=color, linewidth=1.5,
                markersize=6, label=label, zorder=3)
        # linear fit through origin
        if len(xs) >= 2:
            slope = np.polyfit(xs, ys, 1)[0]
            x_fit = np.array([0, max(xs)])
            ax.plot(x_fit, slope * x_fit, color=color, linewidth=0.8,
                    linestyle="--", alpha=0.5, zorder=2,
                    label=f"{label} linear fit (slope={slope:.2f} s/bond)")

    ax.set_xlabel("Number of bonds  $n_{\\mathrm{bonds}}$")
    ax.set_ylabel("Trotter time at saturation (s / layer)")
    ax.set_title(f"Grid scaling at χ={runs_data[0][1]['chi_max']} — trotter time vs bonds")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = f"{out_prefix}_grid.png"
    fig.savefig(path, dpi=dpi)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-layer timing breakdown (trotter + BP) for multiple chi values
# ══════════════════════════════════════════════════════════════════════════════

CHI_COLORS = {50: "#4CAF50", 80: "#FF9800", 120: "#e84040", "other": "#9C27B0"}

def chi_color(chi):
    return CHI_COLORS.get(chi, CHI_COLORS["other"])

def make_breakdown_figure(runs_data, out_prefix, dpi, show):
    """
    Per-layer trotter and BP timing for each chi value in runs_data.
    Trotter: solid line.  BP: dashed line, same color.
    Y-axis log scale (BP spans ~3 orders of magnitude at large chi).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, log in runs_data:
        chi    = log["chi_max"]
        layers = [lay for lay in log["layers"]
                  if lay["layer"] > 0
                  and lay["trotter_ms"] is not None
                  and lay["bp_ms"] is not None]
        if not layers:
            continue

        xs         = [lay["layer"] for lay in layers]
        trotter_ms = [lay["trotter_ms"] for lay in layers]
        bp_ms      = [lay["bp_ms"]      for lay in layers]
        color      = chi_color(chi)

        ax.plot(xs, trotter_ms, color=color, linewidth=1.6,
                linestyle="-",  marker="o", markersize=3,
                label=f"χ={chi} trotter", zorder=3)
        ax.plot(xs, bp_ms,      color=color, linewidth=1.2,
                linestyle="--", marker="s", markersize=3,
                label=f"χ={chi} BP",      zorder=3, alpha=0.8)

    ax.set_xlabel("Trotter layer")
    ax.set_ylabel("Time per layer  (ms)")
    ax.set_yscale("log")
    ax.set_title("Per-layer timing: trotter (—) and BP (- -) for χ ∈ {50, 80, 120}  [MI100]")
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.set_xlim(left=1)
    fig.tight_layout()

    path = f"{out_prefix}_breakdown.png"
    fig.savefig(path, dpi=dpi)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    runs_raw, julia_path, out_prefix, dpi, show = parse_args(sys.argv)

    if not runs_raw:
        print("No log files specified. Run with --help for usage.")
        sys.exit(1)

    # parse all logs
    runs_data = []
    for label, path in runs_raw:
        if not os.path.isfile(path):
            print(f"Warning: file not found: {path}", file=sys.stderr)
            continue
        log = parse_log(path)
        if not log["layers"]:
            print(f"Warning: no data rows parsed from {path}", file=sys.stderr)
            continue
        runs_data.append((label, log))
        grid_str = f"{log['grid_k']}x{log['grid_l']}" if log["grid_k"] else "?"
        print(f"  [{label}] {path}  grid={grid_str}  chi={log['chi_max']}  "
              f"layers={len(log['layers'])}")

    julia_data = None
    julia_label = "Julia χ=50 (reference)"
    if julia_path:
        if not os.path.isfile(julia_path):
            print(f"Warning: Julia file not found: {julia_path}", file=sys.stderr)
        else:
            julia_data = load_julia(julia_path)
            # infer precision tag from filename
            fname = os.path.basename(julia_path).lower()
            prec  = "f64" if "f64" in fname else ("f32" if "f32" in fname else "")
            julia_label = f"Julia {prec} χ=50 (reference)".strip()
            print(f"  [Julia] {julia_path}  points={len(julia_data)}")

    # Detect figure types from the data:
    # - C(t): always (we always have C(t) in logs)
    # - Roofline: always (trotter timing + FLOPs)
    # - Grid scaling: only if multiple grid sizes appear
    grids = {(d["grid_k"], d["grid_l"]) for _, d in runs_data}
    chis  = {d["chi_max"] for _, d in runs_data}
    multi_grid = len(grids) > 1
    multi_chi  = len(chis)  > 1

    print()
    make_ct_figure(runs_data, julia_data, julia_label, out_prefix, dpi, show)
    make_roofline_figure(runs_data, out_prefix, dpi, show)
    if multi_grid:
        make_grid_figure(runs_data, out_prefix, dpi, show)
    else:
        print("Single grid size — skipping grid scaling figure.")
    if multi_chi and not multi_grid:
        make_breakdown_figure(runs_data, out_prefix, dpi, show)
    else:
        print("Skipping breakdown figure (needs multiple chi, single grid).")

if __name__ == "__main__":
    main()
