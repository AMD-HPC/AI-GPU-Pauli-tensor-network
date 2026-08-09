"""
space_estimate.py — GPU memory requirements for 2D PEPS simulation.

Formulae (fitted from F32/F64 measurements on 4×4 grid, χ = 40–80):

  State tensor network for an m×n grid with bond dim χ, physical dim d=2:
    4 corner sites  (degree 2):  4  × d × χ²  elements
    2(m+n-4) edge sites (deg 3): 2(m+n-4) × d × χ³  elements
    (m-2)(n-2) interior (deg 4): (m-2)(n-2) × d × χ⁴  elements

  pre(χ, m, n, T) = k × d × [4χ² + 2(m+n-4)χ³ + (m-2)(n-2)χ⁴] × sizeof(T) / 1e6  MB
    k = 3.82  (empirical: HIP pool holds ~3.82× raw state tensor memory at steady state)

  gate_MB(χ, T) = 7 × d × χ⁴ × sizeof(T) / 1e6  MB   (working mem for one gate)

  n_live_max = (GPU_MB − pre) / gate_MB

Feasibility (steady-state, calibrated on 4×4):
  n_live_max ≥ 5  →  ✓  comfortable
  3 ≤ n_live_max < 5  →  ~  tight (may OOM during transient χ growth)
  1 ≤ n_live_max < 3  →  !  very tight (likely OOM)
  n_live_max < 1   →  ✗  infeasible (pre alone fills GPU)

Time scaling (4×4 reference, dt per Trotter layer):
  dt(χ, F32) ≈ 7.47×10⁻⁵ × χ^3.19  s
  dt(χ, F64) ≈ 1.24 × dt(χ, F32)   (F64/F32 throughput ratio measured ≈ 1.24)
  Note: for grids other than 4×4, dt scales with gate count ∝ m×n (roughly).

Caveats:
  • k=3.82 calibrated on 4×4; assumed to hold for other grid sizes (not yet validated).
  • dt estimate is valid for 4×4 only; other grids scale with number of gates per layer.
  • Transient OOM can occur before steady state if pre growth phase is too steep.

Usage:
  python3 space_estimate.py                          # full k×χ table (4≤k≤10, 35<χ≤100)
  python3 space_estimate.py --grid m n               # single grid, sweep χ
  python3 space_estimate.py --grid m n --chi X       # single (m,n,χ) detail
  python3 space_estimate.py --f32                    # F32 table only
  python3 space_estimate.py --f64                    # F64 table only
  python3 space_estimate.py --gpu GB                 # custom GPU size (default 31.98)
  python3 space_estimate.py --chi 50 60 70 80        # custom χ list
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPU_DEFAULT = 31_980.0   # MB  (31.98 GB AMD GPU)
K           = 3.82       # empirical pool multiplier
D           = 2          # physical dimension (Pauli space)
ALPHA       = 3.19       # dt ∝ χ^α  (fitted from F32 4×4 data)
A_F32       = 7.47e-5    # dt = A × χ^α  [s], F32, 4×4 reference
F32_DT_RATIO = 1.24      # dt(F64) / dt(F32)

F32 = 8                  # bytes per ComplexF32
F64 = 16                 # bytes per ComplexF64

CHI_DEFAULT  = list(range(35, 101, 5))   # 35, 40, 45, ..., 100
K_DEFAULT    = list(range(4, 11))         # 4, 5, ..., 10

# ---------------------------------------------------------------------------
# Core formulae
# ---------------------------------------------------------------------------

def site_counts(m, n):
    """(n_corner, n_edge, n_interior) for an m×n grid."""
    if m < 1 or n < 1:
        raise ValueError(f"Grid must be at least 1×1, got {m}×{n}")
    n_corner   = min(4, 2*m) if m == 1 else (4 if n > 1 else 2*m)
    n_edge     = max(0, 2*(m-2) + 2*(n-2)) if m > 1 and n > 1 else 0
    n_interior = max(0, (m-2)*(n-2))
    # Simpler direct formula for standard m×n (m,n ≥ 2):
    if m >= 2 and n >= 2:
        n_corner   = 4
        n_edge     = 2*(m-2) + 2*(n-2)
        n_interior = (m-2)*(n-2)
    return n_corner, n_edge, n_interior

def state_elements(m, n, chi):
    """Total complex elements in the state tensor network."""
    nc, ne, ni = site_counts(m, n)
    return D * (nc*chi**2 + ne*chi**3 + ni*chi**4)

def pre_mb(m, n, chi, sz):
    """Predicted steady-state pool pre-layer memory (MB)."""
    return K * state_elements(m, n, chi) * sz / 1e6

def gate_mb(chi, sz):
    """Working memory for one gate application (MB)."""
    return 7 * D * chi**4 * sz / 1e6

def n_live_max(m, n, chi, sz, gpu_mb):
    """Maximum concurrent gate workspaces that fit alongside pre."""
    p = pre_mb(m, n, chi, sz)
    g = gate_mb(chi, sz)
    if p >= gpu_mb or g == 0:
        return float('-inf')
    return (gpu_mb - p) / g

def dt_estimate(chi, sz, m=4, n=4):
    """Estimated dt per Trotter layer (s). Valid for 4×4 reference grid."""
    nc, ne, ni = site_counts(m, n)
    nc_ref, ne_ref, ni_ref = site_counts(4, 4)
    gate_scale = (nc*chi**4 + ne*chi**4 + ni*chi**4) / \
                 (nc_ref*chi**4 + ne_ref*chi**4 + ni_ref*chi**4)  # ≈ m*n/16
    ratio = sz / F32
    throughput_ratio = 1.0 if sz == F32 else F32_DT_RATIO
    return A_F32 * chi**ALPHA * gate_scale * throughput_ratio

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def flag(nlive):
    if nlive == float('-inf') or nlive < 1: return '✗'
    if nlive < 3:  return '!'
    if nlive < 5:  return '~'
    return '✓'

def fmt_nlive(nlive):
    if nlive == float('-inf') or nlive < 0:
        return '  ✗  '
    return f'{nlive:5.1f}{flag(nlive)}'

def fmt_pre(mb):
    if mb >= 1000:
        return f'{mb/1024:6.1f}G'
    return f'{mb:6.0f}M'

# ---------------------------------------------------------------------------
# Detail output: single (m, n, chi)
# ---------------------------------------------------------------------------

def print_detail(m, n, chi, gpu_mb):
    nc, ne, ni = site_counts(m, n)
    print(f"\n  Grid: {m}×{n}   χ={chi}   GPU={gpu_mb/1024:.2f} GB")
    print(f"  Sites: {nc} corner (deg2) + {ne} edge (deg3) + {ni} interior (deg4) = {nc+ne+ni} total")
    print()
    hdr = f"  {'Precision':>10}  {'sizeof':>6}  {'state(MB)':>10}  {'pre(MB)':>9}  " \
          f"{'gate(MB)':>9}  {'n_live_max':>10}  {'flag':>4}  {'dt(s)':>8}"
    print(hdr)
    print("  " + "─"*(len(hdr)-2))
    for label, sz in [("F32", F32), ("F64", F64)]:
        se   = state_elements(m, n, chi) * sz / 1e6
        p    = pre_mb(m, n, chi, sz)
        g    = gate_mb(chi, sz)
        nl   = n_live_max(m, n, chi, sz, gpu_mb)
        dt   = dt_estimate(chi, sz, m, n) if (m == 4 and n == 4) else None
        dt_s = f"{dt:8.1f}" if dt is not None else "    n/a "
        nl_s = f"{nl:10.1f}" if nl != float('-inf') else "   —(pre>GPU)"
        print(f"  {label:>10}  {sz:>6}  {se:>10.1f}  {p:>9.1f}  "
              f"{g:>9.1f}  {nl_s:>10}  {flag(nl):>4}  {dt_s}")
    print()
    print("  ✓ n_live≥5 (comfortable)  ~ n_live 3-5 (tight)  "
          "! n_live 1-3 (very tight)  ✗ infeasible")

# ---------------------------------------------------------------------------
# Table: χ rows × k columns (square k×k grids)
# ---------------------------------------------------------------------------

def print_kchi_table(chis, ks, sz, label, gpu_mb, show_pre=False):
    typename = "F32" if sz == F32 else "F64"
    print(f"\n  {typename} — {label}   (GPU = {gpu_mb/1024:.2f} GB,  k = {K})")

    # Header
    col_w = 8
    row_hdr = f"  {'χ':>4}"
    for k in ks:
        row_hdr += f"  {f'{k}×{k}':>{col_w}}"
    sep = "  " + "─" * (len(row_hdr) - 2)
    print(sep)
    print(row_hdr)
    print(sep)

    for chi in chis:
        row = f"  {chi:>4}"
        for k in ks:
            nl = n_live_max(k, k, chi, sz, gpu_mb)
            p  = pre_mb(k, k, chi, sz)
            if show_pre:
                cell = f"{fmt_pre(p):>6}"
            else:
                cell = fmt_nlive(nl)
            row += f"  {cell:>{col_w}}"
        print(row)

    print(sep)

def print_kchi_tables(chis, ks, gpu_mb, precisions, show_pre=False):
    metric = "pre (MB)" if show_pre else "n_live_max  [✓≥5  ~3-5  !1-3  ✗<1]"
    print(f"\n{'═'*72}")
    print(f"  SPACE ESTIMATE — {metric}")
    print(f"{'═'*72}")
    for sz, label in precisions:
        print_kchi_table(chis, ks, sz, label, gpu_mb, show_pre)

# ---------------------------------------------------------------------------
# Single grid: sweep χ
# ---------------------------------------------------------------------------

def print_grid_sweep(m, n, chis, gpu_mb):
    nc, ne, ni = site_counts(m, n)
    print(f"\n  Grid {m}×{n}: {nc} corner + {ne} edge + {ni} interior sites")
    print(f"  pre(χ,T) = {K} × d × "
          f"[{nc}χ² + {ne}χ³ + {ni}χ⁴] × sizeof(T) / 1e6  MB")

    hdr = (f"\n  {'χ':>4}  "
           f"{'pre F32':>9}  {'gate F32':>9}  {'n_live F32':>10}  "
           f"{'pre F64':>9}  {'gate F64':>9}  {'n_live F64':>10}")
    sep = "  " + "─"*(len(hdr)-3)
    print(hdr); print(sep)

    for chi in chis:
        p32 = pre_mb(m, n, chi, F32);  g32 = gate_mb(chi, F32);  nl32 = n_live_max(m, n, chi, F32, gpu_mb)
        p64 = pre_mb(m, n, chi, F64);  g64 = gate_mb(chi, F64);  nl64 = n_live_max(m, n, chi, F64, gpu_mb)
        print(f"  {chi:>4}  "
              f"{p32:>9.0f}  {g32:>9.0f}  {fmt_nlive(nl32):>10}  "
              f"{p64:>9.0f}  {g64:>9.0f}  {fmt_nlive(nl64):>10}")
    print(sep)
    print(f"  ✓ n_live≥5  ~ n_live 3-5  ! n_live 1-3  ✗ n_live<1")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU memory estimate for PEPS simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grid', nargs=2, type=int, metavar=('M','N'),
                        help='Grid size M×N (default: sweep k×k for k=4..10)')
    parser.add_argument('--chi', nargs='+', type=int, metavar='X',
                        help='Bond dimension(s) (default: 35..100 step 5)')
    parser.add_argument('--f32', action='store_true', help='F32 only')
    parser.add_argument('--f64', action='store_true', help='F64 only')
    parser.add_argument('--gpu', type=float, default=31.98, metavar='GB',
                        help='GPU memory in GB (default: 31.98)')
    parser.add_argument('--pre', action='store_true',
                        help='Show pre(MB) instead of n_live_max in k×χ table')
    args = parser.parse_args()

    gpu_mb = args.gpu * 1024
    chis   = args.chi if args.chi else CHI_DEFAULT
    chis   = [c for c in chis if c > 30]

    if args.f32 and not args.f64:
        precisions = [(F32, "ComplexF32")]
    elif args.f64 and not args.f32:
        precisions = [(F64, "ComplexF64")]
    else:
        precisions = [(F32, "ComplexF32"), (F64, "ComplexF64")]

    # Single (m,n,chi) detail
    if args.grid and args.chi and len(args.chi) == 1:
        print_detail(args.grid[0], args.grid[1], args.chi[0], gpu_mb)
        return

    # Single grid, sweep χ
    if args.grid:
        m, n = args.grid
        print_grid_sweep(m, n, chis, gpu_mb)
        return

    # Full k×χ table
    ks = K_DEFAULT
    print_kchi_tables(chis, ks, gpu_mb, precisions, show_pre=args.pre)
    if not args.pre:
        print()
        print_kchi_tables(chis, ks, gpu_mb, precisions, show_pre=True)

if __name__ == "__main__":
    main()
