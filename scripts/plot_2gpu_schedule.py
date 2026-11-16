#!/usr/bin/env python3
"""
plot_2gpu_schedule.py — Timeline diagram of 2-GPU coloring-driven sync protocol.

Shows per-color execution on GPU 0 and GPU 1, with P2P sync points
at boundary-crossing colors.

Usage:
  python3 scripts/plot_2gpu_schedule.py --out /tmp/fig_2gpu_schedule.png
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/fig_2gpu_schedule.png')
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_facecolor('white')

    # Colors
    C_HEVEN  = '#4488cc'  # horizontal even
    C_HODD   = '#66aaee'  # horizontal odd
    C_VEVEN  = '#cc4444'  # vertical even (boundary)
    C_VODD   = '#ee6644'  # vertical odd  (boundary)
    C_BP     = '#44aa66'  # BP
    C_SYNC   = '#ffcc00'  # P2P sync arrow
    C_WAIT   = '#eeeeee'  # waiting

    # Layout
    GPU0_Y = 3.0
    GPU1_Y = 1.5
    LABEL_X = -0.3
    H = 0.7   # block height
    GAP = 0.05

    # Block widths (proportional to cost)
    W = {
        'H': 1.0,   # horizontal color
        'V': 1.2,   # vertical color (boundary)
        'BP': 1.5,  # BP
        'S': 0.3,   # sync gap
    }

    def block(ax, x, y, w, color, label, fontsize=8):
        rect = mpatches.FancyBboxPatch((x+GAP, y-H/2+GAP), w-2*GAP, H-2*GAP,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor='white', lw=1.5)
        ax.add_patch(rect)
        ax.text(x+w/2, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')
        return x + w

    def wait_block(ax, x, y, w):
        rect = mpatches.FancyBboxPatch((x+GAP, y-H/2+GAP), w-2*GAP, H-2*GAP,
            boxstyle="round,pad=0.02", facecolor=C_WAIT, edgecolor='#cccccc', lw=1,
            linestyle='--')
        ax.add_patch(rect)
        ax.text(x+w/2, y, 'wait', ha='center', va='center',
                fontsize=7, color='#999999', style='italic')
        return x + w

    def sync_arrow(ax, x, y0, y1, label):
        ax.annotate('', xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle='->', color=C_SYNC, lw=2.5))
        ax.text(x + 0.1, (y0+y1)/2, label, fontsize=7, color='#aa8800',
                va='center', fontweight='bold')

    # GPU labels
    ax.text(LABEL_X, GPU0_Y, 'GPU 0\n(rows 0-1)', ha='right', va='center',
            fontsize=9, fontweight='bold')
    ax.text(LABEL_X, GPU1_Y, 'GPU 1\n(rows 2)', ha='right', va='center',
            fontsize=9, fontweight='bold')

    # Timeline title
    ax.text(6, 4.2, '2-GPU Trotter step — color-driven sync protocol (3×3 example)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # ── One Trotter step ─────────────────────────────────────────────────────

    x = 0.0

    # H_even: parallel on both GPUs
    block(ax, x, GPU0_Y, W['H'], C_HEVEN, 'H_even\n(0,1)(3,4)')
    block(ax, x, GPU1_Y, W['H'], C_HEVEN, 'H_even\n(6,7)')
    x += W['H']

    # H_odd: parallel
    block(ax, x, GPU0_Y, W['H'], C_HODD, 'H_odd\n(1,2)(4,5)')
    block(ax, x, GPU1_Y, W['H'], C_HODD, 'H_odd\n(7,8)')
    x += W['H']

    # ONE BIDIRECTIONAL SYNC before cut color
    sync_x = x + W['S']/2
    # bidirectional arrow
    ax.annotate('', xy=(sync_x, GPU1_Y + H/2), xytext=(sync_x, GPU0_Y - H/2),
                arrowprops=dict(arrowstyle='<->', color=C_SYNC, lw=2.5))
    ax.text(sync_x + 0.1, (GPU0_Y+GPU1_Y)/2,
            ' SYNC\n site3↔site6\n (boundary)',
            fontsize=7, color='#aa8800', va='center', fontweight='bold')
    x += W['S']

    # V_even (cut color): both GPUs compute cut bond simultaneously
    block(ax, x, GPU0_Y, W['V'], C_VEVEN, 'V_even\n(0,3)(1,4)(2,5)\n+cut bond')
    block(ax, x, GPU1_Y, W['V'], C_VEVEN, 'V_even\n(cut bond)\n→keep own site')
    x += W['V']

    # V_odd: fully parallel, no cut bonds
    block(ax, x, GPU0_Y, W['V'], C_VODD, 'V_odd\n(no cut)')
    block(ax, x, GPU1_Y, W['V'], C_VODD, 'V_odd\n(no cut)')
    x += W['V']

    # BP: sequential then exchange
    block(ax, x, GPU0_Y, W['BP'], C_BP, 'BP\n(rows 0-1)')
    block(ax, x, GPU1_Y, W['BP'], C_BP, 'BP\n(rows 1-2)')
    x += W['BP']

    # BP message exchange
    sync_x = x + W['S']/2
    ax.annotate('', xy=(sync_x, GPU1_Y + H/2), xytext=(sync_x, GPU0_Y - H/2),
                arrowprops=dict(arrowstyle='<->', color='#44aa66', lw=2.5))
    ax.text(sync_x + 0.1, (GPU0_Y + GPU1_Y)/2, ' BP msg\n exchange',
            fontsize=7, color='#227744', va='center', fontweight='bold')
    x += W['S']

    # Next step indicator
    ax.axvline(x + 0.1, color='#999999', lw=1, linestyle=':')
    ax.text(x + 0.3, (GPU0_Y + GPU1_Y)/2, 'next\nstep', fontsize=7,
            color='#999999', va='center', style='italic')

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_HEVEN, label='Horizontal color (parallel)'),
        mpatches.Patch(color=C_HODD,  label='Horizontal color (parallel)'),
        mpatches.Patch(color=C_VEVEN, label='Vertical color — boundary (sequential)'),
        mpatches.Patch(color=C_VODD,  label='Vertical color — boundary (sequential)'),
        mpatches.Patch(color=C_BP,    label='BP convergence'),
        mpatches.Patch(color=C_WAIT,  label='Wait (ghost row ready)'),
        mpatches.Patch(color=C_SYNC,  label='P2P site tensor sync'),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=4,
              fontsize=7, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.08))

    # Boundary row annotation
    ax.annotate('', xy=(0, GPU0_Y - H/2 - 0.15), xytext=(x, GPU0_Y - H/2 - 0.15),
                arrowprops=dict(arrowstyle='<->', color='#aaaaaa', lw=1))
    ax.text((x)/2, GPU0_Y - H/2 - 0.28, 'one Trotter step', ha='center',
            fontsize=8, color='#666666')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
