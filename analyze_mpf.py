"""
analyze_mpf.py — Deep dive analysis of MPF experiment results

Reads main_heavyhex_mpf output and produces:
  1. Per-cycle table: C_nl, OV(gamma), extrapolation, error, monotonicity
  2. Chi growth and timing summary
  3. Failure mode diagnosis
  4. Cross-experiment comparison (when multiple files given)

Usage:
  # Single experiment deep dive
  python3 analyze_mpf.py logs/mpf_experiments/mpf_co4_ct2_np25_nl400_nn150_s10.log

  # Compare all experiments
  python3 analyze_mpf.py logs/mpf_experiments/mpf_*.log

  # Summary table only
  python3 analyze_mpf.py --summary logs/mpf_experiments/mpf_*.log
"""

import sys
import os
import re
import argparse
import numpy as np

GAMMAS    = [1.0, 1.4, 1.8]
MPF_COEFFS = [7.875, -11.250, 4.375]  # for gamma={1.0, 1.4, 1.8}

# ── Parse main_heavyhex_mpf output ────────────────────────────────────────────

def parse_mpf_log(path):
    """Parse main_heavyhex_mpf stdout. Returns dict with metadata and rows."""
    meta = {'file': os.path.basename(path), 'chi_nl': None, 'chi_n': None,
            'cutoff': None, 'center': None, 'n_steps': None}
    rows = []

    with open(path) as f:
        for line in f:
            # Parse header
            m = re.search(r'chi_max=(\d+)', line)
            if m and 'GPU 0' in line: meta['chi_nl'] = int(m.group(1))
            if m and 'GPU 1' in line: meta['chi_n']  = int(m.group(1))
            m = re.search(r'cutoff=([\d.eE+\-]+)', line)
            if m: meta['cutoff'] = float(m.group(1))
            m = re.search(r'center=site(\d+)', line)
            if m: meta['center'] = int(m.group(1))
            m = re.search(r'n_steps=(\d+)', line)
            if m: meta['n_steps'] = int(m.group(1))

            # Parse data rows
            m = re.match(
                r'^\s*(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)'
                r'\s+([\d.eE+\-]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                line)
            if m and int(m.group(1)) > 0:
                cycle  = int(m.group(1))
                C_nl   = float(m.group(2))
                ov     = [float(m.group(3)), float(m.group(4)), float(m.group(5))]
                chi_nl = int(m.group(6))
                chi_n  = int(m.group(7))
                t_nl   = float(m.group(8))
                t_n    = float(m.group(9))
                t_tot  = float(m.group(10))
                extrap = sum(c*o for c,o in zip(MPF_COEFFS, ov))
                err    = abs(C_nl - extrap)

                # Monotonicity of OV in gamma
                if ov[0] < ov[1] < ov[2]:   mono = '↑ mono'
                elif ov[0] > ov[1] > ov[2]: mono = '↓ mono'
                else:                         mono = 'non-mono'

                # Chi saturation
                capped = chi_nl >= (meta['chi_nl'] or 9999)

                # Quality
                if err < 1e-3:   quality = 'excellent'
                elif err < 1e-2: quality = 'good'
                elif err < 5e-2: quality = 'fair'
                else:            quality = 'poor'

                rows.append({
                    'cycle': cycle, 'C_nl': C_nl, 'OV': ov,
                    'chi_nl': chi_nl, 'chi_n': chi_n,
                    't_nl': t_nl, 't_n': t_n, 't_tot': t_tot,
                    'extrap': extrap, 'err': err,
                    'mono': mono, 'capped': capped, 'quality': quality
                })

    return meta, rows

# ── Deep dive: one experiment ─────────────────────────────────────────────────

def deep_dive(path):
    meta, rows = parse_mpf_log(path)
    if not rows:
        print(f"No data in {path}"); return

    print(f"\n{'═'*85}")
    print(f"  DEEP DIVE: {meta['file']}")
    print(f"  chi_nl={meta['chi_nl']}  chi_n={meta['chi_n']}  "
          f"cutoff={meta['cutoff']:.0e}  center={meta['center']}  "
          f"steps={meta['n_steps']}")
    print(f"{'═'*85}")

    print(f"\n{'Cy':>3} {'C_nl':>8} {'OV(1.0)':>8} {'OV(1.4)':>8} {'OV(1.8)':>8} "
          f"{'chi_nl':>7} {'chi_n':>6} {'extrap':>8} {'|err|':>8} {'mono':>8} {'capped':>7} {'quality':>9}")
    print('-'*95)

    for r in rows:
        cap_str = '⚠ cap' if r['capped'] else ''
        print(f"{r['cycle']:>3} {r['C_nl']:>8.5f} {r['OV'][0]:>8.5f} {r['OV'][1]:>8.5f} "
              f"{r['OV'][2]:>8.5f} {r['chi_nl']:>7} {r['chi_n']:>6} "
              f"{r['extrap']:>8.5f} {r['err']:>8.2e} {r['mono']:>8} "
              f"{cap_str:>7} {r['quality']:>9}")

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    good_cycles  = [r['cycle'] for r in rows if r['quality'] in ('excellent','good')]
    capped_at    = next((r['cycle'] for r in rows if r['capped']), None)
    nonmono      = [r['cycle'] for r in rows if r['mono'] == 'non-mono']
    print(f"  Good MPF cycles: {good_cycles}")
    print(f"  Chi saturates:   cycle {capped_at}" if capped_at else "  Chi never saturates")
    print(f"  Non-monotonic:   cycles {nonmono}" if nonmono else "  OV monotonic throughout")

    # Timing
    print(f"\n  TIMING (last cycle): t_nl={rows[-1]['t_nl']:.1f}s  "
          f"t_n={rows[-1]['t_n']:.1f}s  t_tot={rows[-1]['t_tot']:.1f}s")

# ── Summary: compare all experiments ─────────────────────────────────────────

def summary(paths):
    print(f"\n{'═'*100}")
    print(f"  MPF EXPERIMENT SUMMARY ({len(paths)} runs)")
    print(f"{'═'*100}")
    print(f"\n{'Experiment':45} {'c5_err':>8} {'c8_err':>8} {'good_cyc':>9} {'chi_sat':>8} {'score':>6}")
    print('-'*85)

    for path in sorted(paths):
        meta, rows = parse_mpf_log(path)
        if not rows: continue

        name = meta['file'].replace('.log','')
        c5   = next((r for r in rows if r['cycle']==5), None)
        c8   = next((r for r in rows if r['cycle']==8), None)
        good = [r['cycle'] for r in rows if r['quality'] in ('excellent','good')]
        capped_at = next((r['cycle'] for r in rows if r['capped']), '-')

        e5 = f"{c5['err']:.1e}" if c5 else '-'
        e8 = f"{c8['err']:.1e}" if c8 else '-'

        # Score
        if len(good) >= 7:   score = 'A'
        elif len(good) >= 5: score = 'B'
        elif len(good) >= 3: score = 'C'
        else:                 score = 'D'

        print(f"{name:45} {e5:>8} {e8:>8} {str(good):>9} {str(capped_at):>8} {score:>6}")

    print(f"{'═'*100}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MPF experiment analysis")
    parser.add_argument('logs', nargs='+', help='Log file(s) from main_heavyhex_mpf')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary table only (no deep dive)')
    args = parser.parse_args()

    if args.summary or len(args.logs) > 1:
        summary(args.logs)
    if not args.summary:
        for path in args.logs:
            deep_dive(path)

if __name__ == '__main__':
    main()
