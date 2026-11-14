#!/usr/bin/env python3
"""
plot_scan_geometry_16x32.py — Figure: scan center placement on 16x32 grid.

Three panels:
  Left:   Dense single scan (56 centers, blue)
  Middle: Period-6 scan 1 only (18 centers, red)
  Right:  Period-6 both scans (scan1=red, scan2=blue)

Usage:
  python3 scripts/plot_scan_geometry_16x32.py --out /tmp/fig_scan_geometry.pdf
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NX, NY = 32, 16

# Dense: 56 centers (8x8 Voronoi tile [5,10,23,28,33,46,51] tiled 2x4)
dense_sites = [5,13,21,29,34,42,50,58,71,79,87,95,100,108,116,124,
               129,137,145,153,166,174,182,190,195,203,211,219,
               261,269,277,285,290,298,306,314,327,335,343,351,
               356,364,372,380,385,393,401,409,422,430,438,446,
               451,459,467,475]
dense = set(((s-1)//NX, (s-1)%NX) for s in dense_sites)

# Period-6 quincunx
q1 = [(r, c) for r in range(NY) for c in range(NX) if r%6==0 and c%6==0]
q2 = [(r, c) for r in range(NY) for c in range(NX) if r%6==3 and c%6==3]


def draw_panel(ax, centers_dict, title, legend_handles):
    for r in range(NY):
        for c in range(NX):
            ax.plot(c, NY-1-r, '.', color='#dddddd', markersize=2.5, zorder=1)
    for (r, c), color in centers_dict.items():
        ax.plot(c, NY-1-r, 'o', color=color, markersize=6,
                markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    ax.set_xlim(-1, NX)
    ax.set_ylim(-1, NY)
    ax.set_aspect('equal')
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.set_xticklabels(['0','8','16','24','31'], fontsize=7)
    ax.set_yticks([0, 7, 15])
    ax.set_yticklabels(['15','8','0'], fontsize=7)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel('column', fontsize=7)
    ax.legend(handles=legend_handles, fontsize=6, loc='lower right',
              framealpha=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/fig_scan_geometry.pdf')
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle('Seismic scan geometry on $16\\times32$ lattice (512 qubits)',
                 fontsize=10, y=1.02)

    # Panel 1: dense
    draw_panel(axes[0],
               {rc: '#2255aa' for rc in dense},
               f'Dense single scan\n{len(dense)} centers, 81% detection',
               [mpatches.Patch(color='#2255aa', label=f'{len(dense)} centers')])

    # Panel 2: period-6 both scans
    combined = {rc: '#cc2222' for rc in q1}
    combined.update({rc: '#2255aa' for rc in q2})
    draw_panel(axes[1],
               combined,
               f'Period-6 opposing wavefronts\n{len(q1)+len(q2)} centers, 94% detection',
               [mpatches.Patch(color='#cc2222', label=f'Scan 1: {len(q1)}c (rows 0,6,12)'),
                mpatches.Patch(color='#2255aa', label=f'Scan 2: {len(q2)}c (rows 3,9,15)')])

    axes[0].set_ylabel('row', fontsize=7)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
