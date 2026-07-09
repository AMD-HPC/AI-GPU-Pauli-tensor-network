"""
space_estimate_v2.py — GPU memory requirements for 2D PEPS simulation.

Formulae (recalibrated from F32 4×4 measurements at steady state, χ=40-50):

  State tensor network for an m×n grid with bond dim χ, physical dim d=2:
    4 corner sites  (degree 2):  4  × d × χ²  elements
    2(m+n-4) edge sites (deg 3): 2(m+n-4) × d × χ³  elements
    (m-2)(n-2) interior (deg 4): (m-2)(n-2) × d × χ⁴  elements

  pre(χ, m, n, T) = k × d × [4χ² + 2(m+n-4)χ³ + (m-2)(n-2)χ⁴] × sizeof(T) / 1e6  MB
    k = 3.82  (recalibrated from CLEAN code; old bloated code gave k=8.48: HIP pool holds ~8.48× raw state tensor memory at steady state)
    Previous value was k=3.82 — updated from actual 4×4 F32 χ=50 measurements.

  gate_MB(χ, T) = 7 × d × χ⁴ × sizeof(T) / 1e6  MB   (working mem for one gate, unchanged)

  n_live_max = (GPU_MB − pre) / gate_MB

  dt scaling (recalibrated from F32 4×4 steady state, χ=40 and χ=50):
    dt(χ, F32) ≈ 2.310×10⁻⁷ × χ^4.85  s     (α=4.85; cleaner scaling vs old bloated α=8.02)
    dt(χ, F64) ≈ 1.24 × dt(χ, F32)
    NOTE: α≈8 reflects that CholeskyQR dominates at large χ (O(mχ²)=O(χ⁶) for bulk sites)
    plus SVD O(χ⁴·rank) — total scales steeply with χ.

Feasibility:
  n_live_max ≥ 5  →  ✓  comfortable
  3 ≤ n_live_max < 5  →  ~  tight
  1 ≤ n_live_max < 3  →  !  very tight
  n_live_max < 1   →  ✗  infeasible

Usage:
  python3 space_estimate_v2.py                         # full k×χ table
  python3 space_estimate_v2.py --grid m n              # single grid, sweep χ
  python3 space_estimate_v2.py --grid m n --chi X      # single (m,n,χ) detail
  python3 space_estimate_v2.py --f32                   # F32 only
  python3 space_estimate_v2.py --f64                   # F64 only
  python3 space_estimate_v2.py --gpu GB                # custom GPU size (default 31.98)
  python3 space_estimate_v2.py --chi 50 60 70 80       # custom χ list
  python3 space_estimate_v2.py --compare               # show v1 vs v2 predictions
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# Constants — v2 (recalibrated)
# ---------------------------------------------------------------------------

GPU_DEFAULT  = 31_980.0   # MB  (31.98 GB AMD GPU)
K            = 3.82       # recalibrated from clean code (was 8.48 for bloated code)       # empirical pool multiplier (was 3.82 in v1)
D            = 2          # physical dimension
ALPHA        = 4.85       # recalibrated from clean code (was 8.02 for bloated code)       # dt ∝ χ^α (was 3.19 in v1; reflects CholeskyQR + SVD dominance)
A_F32        = 2.310e-7    # recalibrated from clean code (was 1.317e-12)  # dt = A × χ^α [s], F32, 4×4 reference (was 7.47e-5 in v1)
F32_DT_RATIO = 1.24       # dt(F64) / dt(F32)

F32 = 8                   # bytes per ComplexF32
F64 = 16                  # bytes per ComplexF64

CHI_DEFAULT = list(range(35, 101, 5))
K_DEFAULT   = list(range(4, 11))

# ---------------------------------------------------------------------------
# Core formulae
# ---------------------------------------------------------------------------

def site_counts(m, n):
    if m >= 2 and n >= 2:
        return 4, 2*(m-2) + 2*(n-2), (m-2)*(n-2)
    raise ValueError(f"Grid must be at least 2×2, got {m}×{n}")

def state_elements(m, n, chi):
    nc, ne, ni = site_counts(m, n)
    return D * (nc*chi**2 + ne*chi**3 + ni*chi**4)

def pre_mb(m, n, chi, sz):
    return K * state_elements(m, n, chi) * sz / 1e6

def gate_mb(chi, sz):
    return 7 * D * chi**4 * sz / 1e6

def n_live_max(m, n, chi, sz, gpu_mb):
    p = pre_mb(m, n, chi, sz)
    g = gate_mb(chi, sz)
    if p >= gpu_mb or g == 0:
        return float('-inf')
    return (gpu_mb - p) / g

def dt_estimate(chi, sz, m=4, n=4):
    """Estimated dt per Trotter layer (s). Calibrated for 4×4 only."""
    nc, ne, ni = site_counts(m, n)
    nc_ref, ne_ref, ni_ref = site_counts(4, 4)
    gate_scale = (nc + ne + ni) / (nc_ref + ne_ref + ni_ref)
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
# Detail output
# ---------------------------------------------------------------------------

def print_detail(m, n, chi, gpu_mb):
    nc, ne, ni = site_counts(m, n)
    print(f"\n  Grid: {m}×{n}   χ={chi}   GPU={gpu_mb/1024:.2f} GB  [space_estimate_v2: k={K}, α={ALPHA}]")
    print(f"  Sites: {nc} corner + {ne} edge + {ni} interior = {nc+ne+ni} total")
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
        is_ref = (m == 4 and n == 4)
        dt_s = f"{dt_estimate(chi, sz, m, n):8.1f}" if is_ref else "    n/a "
        nl_s = f"{nl:10.1f}" if nl != float('-inf') else "   —(pre>GPU)"
        print(f"  {label:>10}  {sz:>6}  {se:>10.1f}  {p:>9.1f}  "
              f"{g:>9.1f}  {nl_s:>10}  {flag(nl):>4}  {dt_s}")
    print()
    print("  ✓ n_live≥5  ~ 3-5  ! 1-3  ✗ <1")

# ---------------------------------------------------------------------------
# k×χ table
# ---------------------------------------------------------------------------

def print_kchi_table(chis, ks, sz, label, gpu_mb, show_pre=False):
    typename = "F32" if sz == F32 else "F64"
    print(f"\n  {typename} — {label}   (GPU={gpu_mb/1024:.2f} GB, k={K})")
    col_w = 8
    row_hdr = f"  {'χ':>4}"
    for k in ks:
        row_hdr += f"  {f'{k}×{k}':>{col_w}}"
    sep = "  " + "─"*(len(row_hdr)-2)
    print(sep); print(row_hdr); print(sep)
    for chi in chis:
        row = f"  {chi:>4}"
        for k in ks:
            nl = n_live_max(k, k, chi, sz, gpu_mb)
            p  = pre_mb(k, k, chi, sz)
            cell = fmt_pre(p) if show_pre else fmt_nlive(nl)
            row += f"  {cell:>{col_w}}"
        print(row)
    print(sep)

def print_kchi_tables(chis, ks, gpu_mb, precisions, show_pre=False):
    metric = "pre (MB)" if show_pre else "n_live_max  [✓≥5  ~3-5  !1-3  ✗<1]"
    print(f"\n{'═'*72}")
    print(f"  SPACE ESTIMATE v2 — {metric}")
    print(f"{'═'*72}")
    for sz, label in precisions:
        print_kchi_table(chis, ks, sz, label, gpu_mb, show_pre)

# ---------------------------------------------------------------------------
# Single grid sweep
# ---------------------------------------------------------------------------

def print_grid_sweep(m, n, chis, gpu_mb):
    nc, ne, ni = site_counts(m, n)
    print(f"\n  Grid {m}×{n}: {nc} corner + {ne} edge + {ni} interior   [v2: k={K}, α={ALPHA}]")
    hdr = (f"\n  {'χ':>4}  "
           f"{'pre F32':>9}  {'gate F32':>9}  {'n_live F32':>10}  {'dt F32':>8}  "
           f"{'pre F64':>9}  {'gate F64':>9}  {'n_live F64':>10}  {'dt F64':>8}")
    sep = "  " + "─"*(len(hdr)-3)
    print(hdr); print(sep)
    for chi in chis:
        p32  = pre_mb(m, n, chi, F32);   g32  = gate_mb(chi, F32);   nl32 = n_live_max(m, n, chi, F32, gpu_mb)
        p64  = pre_mb(m, n, chi, F64);   g64  = gate_mb(chi, F64);   nl64 = n_live_max(m, n, chi, F64, gpu_mb)
        is_ref = (m == 4 and n == 4)
        dt32 = f"{dt_estimate(chi, F32, m, n):8.1f}" if is_ref else "    n/a "
        dt64 = f"{dt_estimate(chi, F64, m, n):8.1f}" if is_ref else "    n/a "
        print(f"  {chi:>4}  "
              f"{p32:>9.0f}  {g32:>9.0f}  {fmt_nlive(nl32):>10}  {dt32}  "
              f"{p64:>9.0f}  {g64:>9.0f}  {fmt_nlive(nl64):>10}  {dt64}")
    print(sep)
    print(f"  ✓ n_live≥5  ~ 3-5  ! 1-3  ✗ <1")

# ---------------------------------------------------------------------------
# Compare v1 vs v2
# ---------------------------------------------------------------------------

def print_compare(chis, gpu_mb):
    K_V1 = 3.82; A_V1 = 7.47e-5; ALPHA_V1 = 3.19
    print(f"\n  Comparison: v1 (k=3.82, α=3.19) vs v2 (k={K}, α={ALPHA})  [F32, 4×4]")
    hdr = f"\n  {'χ':>4}  {'pre v1':>9}  {'pre v2':>9}  {'meas':>9}  {'dt v1':>8}  {'dt v2':>8}"
    sep = "  " + "─"*(len(hdr)-3)
    print(hdr); print(sep)
    # Measured values (from log)
    measured = {37: (None, 9.16), 40: (None, 9.35), 50: (1590.3, 39.4)}
    for chi in chis:
        se = state_elements(4, 4, chi) * F32 / 1e6
        p1 = K_V1 * se;  p2 = K * se
        dt1 = A_V1 * chi**ALPHA_V1;  dt2 = A_F32 * chi**ALPHA
        m_pre, m_dt = measured.get(chi, (None, None))
        meas_pre = f"{m_pre:9.0f}" if m_pre else "        —"
        print(f"  {chi:>4}  {p1:>9.0f}  {p2:>9.0f}  {meas_pre}  {dt1:>8.1f}  {dt2:>8.1f}")
    print(sep)

# ---------------------------------------------------------------------------
# Heavy-hex topology (heavy_hex_3x3: 52 deg-2, 16 deg-3, no deg-4)
# CppSim workspace uses max_degree=3 so m_max=chi^2 (not chi^3)
# ---------------------------------------------------------------------------

HH_DEG2  = 52    # arm sites (degree 2)
HH_DEG3  = 16    # junction sites (degree 3)
HH_BONDS = 76
HH_D     = 4     # Pauli basis physical dim

def hh_state_mb(chi, bpe=4):
    """State tensor memory (MB) for heavy-hex, Float32 by default."""
    return HH_D * (HH_DEG2 * chi**2 + HH_DEG3 * chi**3) * bpe / 1e6

def hh_msg_mb(chi, bpe=4):
    """BP message memory (MB): 2 per bond, each chi x chi."""
    return 2 * HH_BONDS * chi**2 * bpe / 1e6

def hh_workspace_mb(chi, bpe=4):
    """CppSim workspace (MB) with max_degree=3: m_max=chi^2."""
    m_max = chi**2 + 1
    n_max = chi * HH_D
    large  = m_max * n_max               # 4 large buffers: cv1,cv2,Q1,Q2
    medium = (n_max * HH_D)**2           # 2 medium: RR4, oR4 (+ U,Vt reuse)
    small  = n_max * n_max               # 4 small: R1,R2,A1,A2,gram
    env    = chi * chi                   # 8 env: 4 sqrt_env + 4 inv_sqrt_env
    total  = (4*large + 2*medium + 4*small + 8*env) * bpe / 1e6
    return total

def hh_bp_scratch_mb(chi, bpe=8):
    """Measures BP scratch (float64 by default in CppSim)."""
    return HH_D * chi**2 * bpe / 1e6 * HH_BONDS

def print_heavyhex_sweep(chis, gpu_mb):
    print(f"\n{'═'*70}")
    print(f"  HEAVY-HEX 3×3 SPACE ESTIMATE")
    print(f"  {HH_DEG2} deg-2 sites + {HH_DEG3} deg-3 sites  |  {HH_BONDS} bonds  |  D={HH_D} (Pauli)")
    print(f"  CppSim workspace: max_degree=3 → m_max=χ²  |  GPU={gpu_mb/1024:.1f} GB")
    print(f"{'═'*70}")
    hdr = (f"\n  {'χ':>5}  {'state':>9}  {'msg':>7}  {'workspace':>11}  "
           f"{'bp_scratch':>11}  {'total':>9}  {'total GB':>9}  {'fits?':>6}")
    sep = "  " + "─"*(len(hdr)-3)
    print(hdr); print(sep)
    for chi in chis:
        s   = hh_state_mb(chi)
        m   = hh_msg_mb(chi)
        ws  = hh_workspace_mb(chi)
        bp  = hh_bp_scratch_mb(chi)
        tot = s + m + ws + bp
        ok  = "✓ OK" if tot < gpu_mb else ("~ tight" if tot < gpu_mb*1.1 else "✗ OOM")
        print(f"  {chi:>5}  {s:>9.0f}  {m:>7.0f}  {ws:>11.0f}  "
              f"{bp:>11.0f}  {tot:>9.0f}  {tot/1024:>9.1f}  {ok:>6}")
    print(sep)
    print(f"  Units: MB.  bp_scratch = Measures float64 buffer.")
    print(f"  Max chi for MI100 (32GB): see ✓ rows above.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU memory estimate for PEPS simulation (v2, recalibrated).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grid', nargs=2, type=int, metavar=('M','N'))
    parser.add_argument('--chi', nargs='+', type=int, metavar='X')
    parser.add_argument('--f32', action='store_true')
    parser.add_argument('--f64', action='store_true')
    parser.add_argument('--gpu', type=float, default=31.98, metavar='GB')
    parser.add_argument('--pre', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--heavy-hex', action='store_true',
                        help='Heavy-hex 3x3 topology (52 deg-2, 16 deg-3, 76 bonds, D=4)')
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

    if args.compare:
        print_compare(chis, gpu_mb)
        return

    if args.heavy_hex:
        hh_chis = args.chi if args.chi else list(range(50, 601, 50))
        print_heavyhex_sweep(hh_chis, gpu_mb)
        return

    if args.grid and args.chi and len(args.chi) == 1:
        print_detail(args.grid[0], args.grid[1], args.chi[0], gpu_mb)
        return

    if args.grid:
        m, n = args.grid
        print_grid_sweep(m, n, chis, gpu_mb)
        return

    ks = K_DEFAULT
    print_kchi_tables(chis, ks, gpu_mb, precisions, args.pre)
    if not args.pre:
        print()
        print_kchi_tables(chis, ks, gpu_mb, precisions, show_pre=True)

if __name__ == "__main__":
    main()
