#!/usr/bin/env python3
"""
make_mi_c_lightcone.py — Full-page light-cone figure for the GPU-C appendix.

Layout  (one page per figure, ~200 dpi, light theme):
  Row 0-1 : 2 × 4 heatmap panels — n(r,t) at 8 selected Trotter layers
  Row 2   : C(t) panel spanning full width with layer markers

Frame selection uses sqrt-spacing to oversample the fast early phase
(tight diamond) while still covering the late scrambling regime.

Usage:
    python Paper/make_mi_c_lightcone.py LOG -o OUTPUT.png [--layers L1 L2 ...]

Examples:
    python Paper/make_mi_c_lightcone.py \\
        logs/mi355x_20260519/10x10_chi050_n60_all.log \\
        -o Paper/fig_lightcone_10x10_chi050.png

    python Paper/make_mi_c_lightcone.py \\
        logs/mi355x_20260519/11x11_chi100_n70_all.log \\
        -o Paper/fig_lightcone_11x11_chi100.png
"""

import argparse, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

N_FRAMES = 8    # 2 rows × 4 columns


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_log(path):
    """Return (header, layers_list) where each entry has layer, ct, grid."""
    header = {}
    layers = []
    text   = Path(path).read_text(errors='replace')

    m = re.search(r'Grid:\s*(\d+)x(\d+)', text)
    if not m:
        sys.exit(f'Cannot find Grid: NxM in {path}')
    nx, ny = int(m.group(1)), int(m.group(2))
    header['nx'] = nx
    header['ny'] = ny

    m = re.search(r'chi_max=(\d+)', text)
    header['chi_max'] = int(m.group(1)) if m else '?'

    m = re.search(r'QR:\s*(\w+)', text)
    header['qr'] = m.group(1) if m else 'unknown'

    m = re.search(r'Steps:\s*(\d+)', text)
    header['n_steps'] = int(m.group(1)) if m else '?'

    m = re.search(r'Center site: \((\d+),(\d+)\)', text)
    header['center'] = (int(m.group(1)), int(m.group(2))) if m else (ny//2, nx//2)

    # Match every layer data line then look for the following n(r,t) block
    _num = r'(?:[0-9.eE+\-]+|inf|nan|-inf)'
    row_pat = re.compile(
        r'^\s*(\d+)\s+\d+\s+\d+\s+([0-9.eE+\-]+)',
        re.MULTILINE,
    )
    grid_pat = re.compile(
        r'n\(r,t\) \[non-identity Pauli weight\]:(.*?)'
        r'(?=\n\s*\[Trotter\]|\n\s*\d+\s+\d+|\Z)',
        re.DOTALL,
    )

    for rm in row_pat.finditer(text):
        layer = int(rm.group(1))
        if layer == 0:
            continue
        ct  = float(rm.group(2))
        gm  = grid_pat.search(text, rm.end())
        if gm is None:
            layers.append(dict(layer=layer, ct=ct, grid=None))
            continue
        nums = re.findall(r'-?[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?', gm.group(1))
        if len(nums) != nx * ny:
            layers.append(dict(layer=layer, ct=ct, grid=None))
            continue
        grid = np.array([float(x) for x in nums]).reshape(ny, nx)
        layers.append(dict(layer=layer, ct=ct, grid=grid))

    return header, layers


# ── Frame selection ───────────────────────────────────────────────────────────

def pick_frames(layers, n=N_FRAMES, forced=None):
    """
    sqrt-spaced selection from layers that have a grid.
    forced: explicit list of layer indices (overrides auto selection).
    """
    with_grid = [l for l in layers if l['grid'] is not None]
    if not with_grid:
        sys.exit('No spatial frames found in log.')

    if forced:
        by_layer = {l['layer']: l for l in with_grid}
        result   = [by_layer[f] for f in forced if f in by_layer]
        if result:
            return result[:n]
        print('Warning: none of the forced layers have grids; using auto.')

    total = len(with_grid)
    if total <= n:
        return with_grid

    # sqrt spacing — oversample early phase
    raw = np.linspace(0, np.sqrt(total - 1), n) ** 2
    idx = np.unique(np.clip(np.round(raw).astype(int), 0, total - 1))
    # Always include last frame
    if idx[-1] != total - 1:
        idx[-1] = total - 1
    return [with_grid[i] for i in idx[:n]]


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(log_path, out_path, forced_layers=None, vmin=1e-5, dpi=200):
    header, layers = parse_log(log_path)

    nx     = header['nx']
    ny     = header['ny']
    chi    = header['chi_max']
    qr_raw = header['qr']
    center = header['center']     # (row, col) in grid coords
    n_steps= header['n_steps']

    qr_label = {'householder': 'Householder', 'adaptive': 'Adaptive'}.get(
        qr_raw.lower(), qr_raw.capitalize())

    frames = pick_frames(layers, n=N_FRAMES, forced=forced_layers)

    # ── Colour scale (global log norm) ────────────────────────────────────────
    all_vals = np.concatenate([f['grid'].ravel() for f in frames])
    vmax     = float(np.nanmax(all_vals))
    vmin_eff = max(vmin, vmax * 1e-4)
    norm     = LogNorm(vmin=vmin_eff, vmax=vmax, clip=True)
    cmap     = 'inferno'

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row 0-1: 4 panels each; Row 2: C(t) full width
    fig = plt.figure(figsize=(17, 13))
    gs  = fig.add_gridspec(
        3, 4,
        height_ratios=[4, 4, 2.6],
        hspace=0.42, wspace=0.06,
        left=0.05, right=0.89, top=0.91, bottom=0.07,
    )

    im_ref = None
    for idx, frame in enumerate(frames):
        row = idx // 4
        col = idx % 4
        ax  = fig.add_subplot(gs[row, col])

        im = ax.imshow(
            frame['grid'], origin='upper',
            cmap=cmap, norm=norm,
            aspect='equal', interpolation='nearest',
        )
        im_ref = im

        t = frame['layer'] * 0.1
        ax.set_title(
            f'$t = {t:.1f}$   (L = {frame["layer"]})',
            fontsize=10, pad=4,
        )
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

        # White cross at center site (center is (row,col) → imshow is (col,row))
        cx_col, cx_row = center[1], center[0]
        ax.plot(cx_col, cx_row, 'w+', ms=8, mew=2.0, alpha=0.80, zorder=5)

        # C(t) annotation below each panel
        ax.set_xlabel(f'$C(t) = {frame["ct"]:.4g}$', fontsize=8.5, labelpad=3)

    # Shared colour bar
    cbar_ax = fig.add_axes([0.905, 0.36, 0.016, 0.55])
    cb = fig.colorbar(im_ref, cax=cbar_ax)
    cb.set_label(r'$n(\mathbf{r},t)$  [non-identity Pauli weight]', fontsize=10)
    cb.ax.tick_params(labelsize=8)

    # ── C(t) panel ────────────────────────────────────────────────────────────
    ax_ct = fig.add_subplot(gs[2, :])

    all_layers_with_grid = [l for l in layers if l['grid'] is not None]
    t_all  = [l['layer'] * 0.1 for l in all_layers_with_grid]
    ct_all = [abs(l['ct'])     for l in all_layers_with_grid]

    ax_ct.semilogy(t_all, ct_all, color='#1a9850', lw=2.0, zorder=3,
                   label=r'$|C(t)|$')
    ax_ct.set_xlim(left=0)

    for frame in frames:
        ax_ct.axvline(frame['layer'] * 0.1,
                      color='#555555', lw=0.7, ls=':', alpha=0.6, zorder=2)

    ax_ct.set_xlabel(r'Time  $t = n\,\delta t$  ($\delta t = 0.1$)', fontsize=11)
    ax_ct.set_ylabel(r'$|C(t)|$', fontsize=11)
    ax_ct.grid(True, which='both', alpha=0.18, lw=0.5)
    ax_ct.grid(True, which='major', alpha=0.35, lw=0.7)
    ax_ct.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax_ct.tick_params(labelsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        rf'Operator weight $n(\mathbf{{r}},t)$ — '
        rf'${nx}\times{ny}$ lattice, $\chi_{{\mathrm{{max}}}}={chi}$, {qr_label} QR'
        '\n'
        rf'GPU-C (HBM3e, 192+\,GB) — {n_steps} Trotter steps, $\delta t = 0.1$',
        fontsize=13, y=0.99,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'Saved  {out_path}')
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('log',   help='CppSim log file with --measure=all data')
    ap.add_argument('-o', '--output', default=None,
                    help='Output PNG path (default: auto-named in Paper/)')
    ap.add_argument('--layers', nargs='+', type=int, default=None,
                    help='Explicit Trotter layer indices to show (default: auto 8)')
    ap.add_argument('--vmin', type=float, default=1e-5,
                    help='Log-scale floor for colour map (default: 1e-5)')
    ap.add_argument('--dpi',  type=int,   default=200)
    args = ap.parse_args()

    log = Path(args.log)
    if not log.exists():
        sys.exit(f'Log not found: {log}')

    if args.output:
        out = Path(args.output)
    else:
        m = re.match(r'(\d+x\d+)_chi(\d+)', log.stem)
        tag = f'{m.group(1)}_chi{m.group(2)}' if m else log.stem
        out = Path(__file__).parent / f'fig_lightcone_{tag}.png'

    print(f'Log  : {log}')
    print(f'Out  : {out}')
    make_figure(log, out, forced_layers=args.layers, vmin=args.vmin, dpi=args.dpi)


if __name__ == '__main__':
    main()
