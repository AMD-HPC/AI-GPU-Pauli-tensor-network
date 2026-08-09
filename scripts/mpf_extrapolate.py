"""
mpf_extrapolate.py — Multi-Product Formula (MPF) extrapolation

Computes MPF coefficients analytically from gamma values and applies them
to observables (C(t) or per-site Pauli weights) to extrapolate to zero noise.

The MPF system cancels noise up to (n_gamma-1)-th order and extrapolates to gamma=0:
  Σ c_i = 1                    (extrapolation to zero noise)
  Σ c_i * gamma_i^k = 0        for k = 1, ..., n_gamma-1  (noise cancellation)

For gamma = {1.0, 1.4, 1.8} this gives a unique 3×3 linear system.

Usage:
  # From main_heavyhex_mpf output (C(t) extrapolation):
  python3 mpf_extrapolate.py --data mpf_output.dat --gammas 1.0 1.4 1.8

  # With spatial Pauli weight data:
  python3 mpf_extrapolate.py --spatial \\
      --nl  logs/lightcone_nl.dat \\
      --n10 logs/lightcone_n10.dat \\
      --n14 logs/lightcone_n14.dat \\
      --n18 logs/lightcone_n18.dat \\
      --gammas 1.0 1.4 1.8

Input format for --data (from main_heavyhex_mpf stdout):
  # cycle  C_nl  OV(1.0)  OV(1.4)  OV(1.8)  chi_nl  chi_n ...
  1  0.886203  0.886398  0.886475  0.886553  4  4  ...

Output:
  cycle  C_nl  OV_extrap  |C_nl - OV_extrap|  quality
"""

import argparse
import sys
import re
import os
import numpy as np

# ── Compute MPF coefficients ────────────────────────────────────────────────

def mpf_coefficients(gammas):
    """
    Compute MPF coefficients analytically.

    Solves the Vandermonde-like system:
      A @ c = b
    where A[i,j] = gamma_j^i, b = [1, 0, 0, ...]

    Returns: array c of shape (n_gammas,)
    """
    n = len(gammas)
    A = np.array([[g**i for g in gammas] for i in range(n)], dtype=float)
    b = np.zeros(n); b[0] = 1.0
    c = np.linalg.solve(A, b)

    # Validate
    assert abs(sum(c) - 1.0) < 1e-10, "coefficients don't sum to 1"
    for k in range(1, n):
        assert abs(sum(c[i]*gammas[i]**k for i in range(n))) < 1e-10, \
            f"k={k} noise cancellation failed"

    return c

def print_coefficients(gammas, coeffs):
    print(f"\nMPF coefficients for gamma={list(gammas)}:")
    for g, c in zip(gammas, coeffs):
        print(f"  c(gamma={g:.1f}) = {c:+.6f}")
    print(f"  sum = {sum(coeffs):.8f}  (should be 1.0)")
    for k in range(1, len(gammas)):
        val = sum(c*g**k for c,g in zip(coeffs,gammas))
        print(f"  sum c*gamma^{k} = {val:.2e}  (should be 0)")

# ── Parse main_heavyhex_mpf output ──────────────────────────────────────────

def parse_mpf_output(path):
    """
    Parse stdout from main_heavyhex_mpf.
    Returns list of dicts: {cycle, C_nl, OV: [ov1, ov2, ov3], chi_nl, chi_n}
    """
    rows = []
    with open(path) as f:
        for line in f:
            # Match data lines: integer followed by floats
            m = re.match(
                r'^\s*(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)',
                line)
            if m and int(m.group(1)) > 0:
                rows.append({
                    'cycle': int(m.group(1)),
                    'C_nl':  float(m.group(2)),
                    'OV':   [float(m.group(3)), float(m.group(4)), float(m.group(5))],
                })
    return rows

# ── Parse spatial lightcone data ─────────────────────────────────────────────

def parse_spatial(path):
    """
    Parse spatial data from main_heavyhex --spatial=file.
    Returns dict: {cycle: array[n_sites]}
    """
    cycles = {}
    current_cycle = None
    current = {}
    n_sites = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('# heavy_hex'):
                for part in line.split():
                    if part.startswith('n_sites='):
                        n_sites = int(part.split('=')[1])
            elif line.startswith('# cycle'):
                parts = line.split()
                current_cycle = int(parts[2])
                current = {}
            elif line.startswith('#'):
                continue
            elif current_cycle is not None:
                sid, w = int(line.split()[0]), float(line.split()[1])
                current[sid] = w
                if n_sites and len(current) == n_sites:
                    cycles[current_cycle] = np.array([current.get(i,0.) for i in range(n_sites)])
    return cycles

# ── C(t) extrapolation ───────────────────────────────────────────────────────

def extrapolate_ct(rows, coeffs, gammas):
    print(f"\n{'═'*70}")
    print(f"  C(t) MPF EXTRAPOLATION  (gamma={list(gammas)})")
    print(f"{'═'*70}")
    print(f"  {'Cycle':>5}  {'C_nl':>10}  {'OV_extrap':>10}  {'|error|':>10}  {'quality':>8}")
    print(f"  {'-----':>5}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'--------':>8}")

    for row in rows:
        cycle  = row['cycle']
        C_nl   = row['C_nl']
        OV     = row['OV']
        extrap = sum(c*ov for c,ov in zip(coeffs, OV))
        err    = abs(C_nl - extrap)

        # Quality: how well does MPF extrapolation recover noiseless?
        if err < 1e-3:   quality = "excellent"
        elif err < 1e-2: quality = "good"
        elif err < 5e-2: quality = "fair"
        else:            quality = "poor"

        print(f"  {cycle:>5}  {C_nl:>10.6f}  {extrap:>10.6f}  {err:>10.2e}  {quality:>8}")

    print(f"{'═'*70}")

# ── Spatial MPF extrapolation ─────────────────────────────────────────────────

def extrapolate_spatial(spatial_nl, spatial_n, coeffs, gammas, out_dir=None):
    """
    Apply MPF coefficients site-by-site to Pauli weight data.
    spatial_nl: dict {cycle: array[n_sites]} — noiseless
    spatial_n:  list of dicts {cycle: array[n_sites]} — one per gamma
    """
    print(f"\n{'═'*70}")
    print(f"  SPATIAL MPF EXTRAPOLATION  (gamma={list(gammas)})")
    print(f"{'═'*70}")

    common_cycles = sorted(set(spatial_nl.keys()) &
                           set.intersection(*[set(d.keys()) for d in spatial_n]))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for cycle in common_cycles:
        w_nl    = spatial_nl[cycle]
        w_n     = [d[cycle] for d in spatial_n]
        w_extrap = sum(c*w for c,w in zip(coeffs, w_n))

        err_per_site = np.abs(w_nl - w_extrap)
        max_err = err_per_site.max()
        mean_err = err_per_site.mean()

        print(f"  Cycle {cycle:2d}: max|error|={max_err:.2e}  mean|error|={mean_err:.2e}")

        if out_dir:
            fname = os.path.join(out_dir, f"mpf_spatial_cycle{cycle:02d}.dat")
            with open(fname, 'w') as f:
                f.write(f"# MPF spatial extrapolation  cycle={cycle}  gamma={list(gammas)}\n")
                f.write(f"# site_id  w_nl  w_extrap  w_n(1.0)  w_n(1.4)  w_n(1.8)  error\n")
                for i, (wn, we, *wns) in enumerate(zip(w_nl, w_extrap, *w_n)):
                    f.write(f"{i}  {wn:.6f}  {we:.6f}  "
                            f"  ".join(f"{w:.6f}" for w in wns) +
                            f"  {abs(wn-we):.2e}\n")

    print(f"{'═'*70}")
    if out_dir:
        print(f"  Spatial data written to: {out_dir}/")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MPF extrapolation: compute coefficients and apply to observables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('--gammas', nargs='+', type=float, default=[1.0, 1.4, 1.8],
                        help='Gamma values (default: 1.0 1.4 1.8)')
    parser.add_argument('--data', type=str, default=None,
                        help='main_heavyhex_mpf output file for C(t) extrapolation')
    parser.add_argument('--spatial', action='store_true',
                        help='Perform spatial Pauli weight extrapolation')
    parser.add_argument('--nl',  type=str, help='Noiseless spatial data file')
    parser.add_argument('--n10', type=str, help='Noisy gamma=1.0 spatial data file')
    parser.add_argument('--n14', type=str, help='Noisy gamma=1.4 spatial data file')
    parser.add_argument('--n18', type=str, help='Noisy gamma=1.8 spatial data file')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory for spatial MPF data')
    args = parser.parse_args()

    gammas = np.array(args.gammas)
    coeffs = mpf_coefficients(gammas)
    print_coefficients(gammas, coeffs)

    if args.data:
        rows = parse_mpf_output(args.data)
        if not rows:
            print(f"ERROR: no data found in {args.data}")
            sys.exit(1)
        extrapolate_ct(rows, coeffs, gammas)

    if args.spatial:
        missing = [f for f,v in [('--nl',args.nl),('--n10',args.n10),
                                  ('--n14',args.n14),('--n18',args.n18)] if not v]
        if missing:
            print(f"ERROR: spatial mode requires {', '.join(missing)}")
            sys.exit(1)
        spatial_nl = parse_spatial(args.nl)
        spatial_n  = [parse_spatial(args.n10),
                      parse_spatial(args.n14),
                      parse_spatial(args.n18)]
        extrapolate_spatial(spatial_nl, spatial_n, coeffs, gammas, args.out)

    if not args.data and not args.spatial:
        print("\nNo input specified. Use --data or --spatial. See --help.")

if __name__ == '__main__':
    main()
