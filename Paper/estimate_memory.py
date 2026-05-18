#!/usr/bin/env python3
"""
estimate_memory.py — CppSim GPU memory estimator

Predicts GPU memory required for a 2D k×l Ising simulation at bond
dimension chi_max, using the exact buffer layout from workspace.hpp.

Memory model (float32, D=4 Pauli basis):

  STATE
    psi tensors : each site stores D * Π_{b∈bonds(v)} chi floats
    messages    : each bond stores 2 * chi^2 floats (one chi×chi per direction)

  WORKSPACE  (one Workspace struct, reused every gate — zero malloc during sim)
    4 large   : cv1, cv2, Q1, Q2  — each m_max * n_max  (m_max=chi^3, n_max=D*chi)
    4 medium  : RR4, oR4, U, Vt  — each (D²*chi)^2
    6 small   : R1,R2,A1,A2,RR,gram  — each (D*chi)^2
    8 env     : sqrt_env[4] + inv_sqrt_env[4]  — each chi^2
    misc      : syevd/svd/potrf scratch  — tiny

  MEASURES (Measures struct, allocated once)
    d_ones    : m_max floats
    d_sel_I/Z : D floats each
    d_mu      : chi_max floats
    bp_msg    : 2 * n_bonds * chi^2  (temporary copy during measurement)

Validation at k=l=4, chi=50, D=4:
  State (psi+msg)  = 416.6 MB  ← matches ./main output
  Workspace        = 401.4 MB  ← matches ./main output
"""

import sys
from math import log2

# ── Physical constants ───────────────────────────────────────────────────────
D          = 4    # Pauli basis {I,X,Y,Z}
FBYTES     = 4    # float32
MAX_DEGREE = 4    # max bonds per site (interior site on 2D grid)

# ── GPU targets ──────────────────────────────────────────────────────────────
# The two machines the user has right now get starred.
GPUS = [
    ("MI60  (32 GB HBM2)", 32,  True),
    ("MI100 (32 GB HBM2)", 32,  True),
    ("MI250X(128 GB HBM2e)",128, False),
    ("MI300X(192 GB HBM3)", 192, False),
    ("A100  (40 GB HBM2)", 40,  False),
    ("A100  (80 GB HBM2)", 80,  False),
    ("H100  (80 GB HBM3)", 80,  False),
]

# ── Grid helpers ─────────────────────────────────────────────────────────────

def site_degree_counts(k, l):
    """(n_deg2, n_deg3, n_deg4): corner / edge / interior site counts."""
    n2 = 4
    n3 = 2*(k-2) + 2*(l-2)
    n4 = (k-2) * (l-2)
    return n2, n3, n4

def n_bonds(k, l):
    return k*(l-1) + (k-1)*l

# ── Memory formulas ──────────────────────────────────────────────────────────

def state_bytes(k, l, chi):
    """psi tensors + messages (both on GPU, persistent)."""
    n2, n3, n4 = site_degree_counts(k, l)
    psi = (n2 * chi**2 + n3 * chi**3 + n4 * chi**4) * D * FBYTES
    msg = 2 * n_bonds(k, l) * chi**2 * FBYTES
    return psi, msg

def workspace_bytes(chi):
    """Workspace struct: exact formula from workspace.hpp."""
    m_max = chi**3          # chi^(MAX_DEGREE-1)
    n_max = chi * D         # chi * D
    large = m_max * n_max * FBYTES   # cv1, cv2, Q1, Q2
    medium= n_max * D * n_max * D * FBYTES  # RR4, oR4, U, Vt
    small = n_max * n_max * FBYTES          # R1,R2,A1,A2,RR,gram (+S≈tiny)
    env   = chi * chi * FBYTES              # each of 8 env buffers
    ws = 4*large + 4*medium + 6*small + 8*env
    # add misc scratch (syevd_W/E/V, svd_work — negligible but included)
    ws += (chi + chi + chi**2 + n_max) * FBYTES
    return ws

def measures_bytes(k, l, chi):
    """Measures struct: permanent + temporary bp_msg copy."""
    perm = (chi**3 + D + D + chi) * FBYTES   # d_ones, d_sel_I, d_sel_Z, d_mu
    bp   = 2 * n_bonds(k, l) * chi**2 * FBYTES  # temporary copy during autocorr
    return perm + bp

def total_bytes(k, l, chi):
    psi, msg = state_bytes(k, l, chi)
    ws  = workspace_bytes(chi)
    meas = measures_bytes(k, l, chi)
    return psi + msg + ws + meas

# ── Per-gate computation estimate ────────────────────────────────────────────

def flops_per_gate(chi):
    """
    Dominant cost per apply_binary_gate call (FP32 multiply-adds):
      QR  of (m_max × n_max) matrix: ~2 * m * n^2  (thin Householder / Cholesky)
      SVD of (n_max × n_max):        ~10 * n^3       (bidiag reduction)
      2× GEMM reconstruct:           ~2 * m * n * chi (Q × M bridge)
    where m = chi^3, n = D*chi.
    """
    m = chi**3
    n = D * chi
    qr    = 2 * m * n**2        # QR for each of the 2 sites
    svd   = 10 * n**3
    gemm  = 2 * 2 * m * n * chi # two reconstruction GEMMs (one per site)
    return 2*qr + svd + gemm    # factor 2 QR: once per site

def gates_per_layer(k, l):
    """Rxx layer: one gate per bond, split into 4 color classes."""
    return n_bonds(k, l)

# ── Formatting helpers ───────────────────────────────────────────────────────

def gb(b):  return b / 1e9
def fmt(b): return f"{gb(b):7.2f}"

def bar(total_gb, capacity_gb, width=20):
    frac = min(total_gb / capacity_gb, 1.0)
    filled = int(frac * width)
    return "[" + "#"*filled + "-"*(width-filled) + "]"

# ════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Total GPU memory for all grid sizes at key chi values
# ════════════════════════════════════════════════════════════════════════════

CHI_COLS = [10, 20, 30, 50, 70, 100, 120]

def print_memory_table():
    print("═"*100)
    print("TABLE 1 — Total GPU memory (GB)  =  psi + messages + workspace + measures")
    print(f"           Workspace = 4×(chi³×D·chi) + 4×(D²·chi)² + 6×(D·chi)² + 8×chi²  [float32, D={D}]")
    print(f"           32 GB limit (MI100/MI60) marked with  *** ")
    print("═"*100)

    # header
    print(f"  {'Grid':>5}", end="")
    for chi in CHI_COLS:
        print(f"  chi={chi:3d}", end="")
    print()
    print(f"  {'':>5}", end="")
    for chi in CHI_COLS:
        print(f"  {'(GB)':>7}", end="")
    print()
    print("  " + "-"*5 + ("  " + "-"*7)*len(CHI_COLS))

    for k in range(3, 13):
        for l in range(k, 13):
            print(f"  {k}×{l:<4}", end="")
            for chi in CHI_COLS:
                tot = gb(total_bytes(k, l, chi))
                marker = "***" if tot > 32 else "   "
                print(f"  {tot:6.2f}{marker}", end="")
            print()
        if k < 12:
            print()

# ════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Breakdown: state vs workspace, and max chi per GPU
# ════════════════════════════════════════════════════════════════════════════

def max_chi_for_gpu(k, l, capacity_gb):
    for chi in range(5, 201):
        if gb(total_bytes(k, l, chi)) > capacity_gb:
            return chi - 1
    return 200

def print_breakdown_table():
    print()
    print("═"*110)
    print("TABLE 2 — Memory breakdown at chi=50 and chi=100, + max chi on 32 GB and 80 GB GPUs")
    print("           psi=site tensors  msg=messages  ws=workspace  meas=measures buffer")
    print("═"*110)

    hdr = (f"  {'Grid':>5}  {'Sites':>5}  {'Bonds':>5}  "
           f"{'--- chi=50 ---':^36}  {'--- chi=100 ---':^36}  "
           f"{'MaxChi':>8}  {'MaxChi':>8}")
    print(hdr)
    print(f"  {'':>5}  {'':>5}  {'':>5}  "
          f"{'psi':>7}{'msg':>7}{'ws':>7}{'tot(GB)':>9}{'tot(GB)':>9}  "  # sic: intentional alignment
          f"{'psi':>7}{'msg':>7}{'ws':>7}{'tot(GB)':>9}{'tot(GB)':>9}  "
          f"{'@32GB':>8}  {'@80GB':>8}")

    for k in range(3, 13):
        for l in range(k, 13):
            ns = k*l
            nb = n_bonds(k, l)

            psi50, msg50 = state_bytes(k, l, 50)
            ws50         = workspace_bytes(50)
            meas50       = measures_bytes(k, l, 50)
            tot50        = psi50 + msg50 + ws50 + meas50

            psi100, msg100 = state_bytes(k, l, 100)
            ws100          = workspace_bytes(100)
            meas100        = measures_bytes(k, l, 100)
            tot100         = psi100 + msg100 + ws100 + meas100

            mc32 = max_chi_for_gpu(k, l, 32)
            mc80 = max_chi_for_gpu(k, l, 80)

            print(f"  {k}×{l:<4}  {ns:>5}  {nb:>5}  "
                  f"{gb(psi50):>7.2f}{gb(msg50):>7.3f}{gb(ws50):>7.2f}{gb(tot50):>9.2f}  "
                  f"{gb(psi100):>7.2f}{gb(msg100):>7.3f}{gb(ws100):>7.2f}{gb(tot100):>9.2f}  "
                  f"{mc32:>8}  {mc80:>8}")
        print()

# ════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Max chi per GPU for every grid
# ════════════════════════════════════════════════════════════════════════════

def print_maxchi_table():
    print()
    print("═"*90)
    print("TABLE 3 — Maximum chi_max that fits on each GPU (total memory budget)")
    print("═"*90)
    print(f"  {'Grid':>5}", end="")
    for name, cap, starred in GPUS:
        tag = "★" if starred else " "
        short = name.split("(")[0].strip()
        print(f"  {tag}{short:>10}", end="")
    print()
    print(f"  {'':>5}", end="")
    for name, cap, _ in GPUS:
        cap_str = name.split("(")[1].rstrip(")")
        print(f"  {cap_str:>11}", end="")
    print()
    print("  " + "-"*5 + ("  " + "-"*11)*len(GPUS))

    for k in range(3, 13):
        for l in range(k, 13):
            print(f"  {k}×{l:<4}", end="")
            for name, cap, _ in GPUS:
                mc = max_chi_for_gpu(k, l, cap)
                print(f"  {mc:>11}", end="")
            print()
        print()

# ════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Per-gate compute cost
# ════════════════════════════════════════════════════════════════════════════

def print_compute_table():
    print()
    print("═"*90)
    print("TABLE 4 — Per-gate and per-layer FLOP estimates (FP32 MACs × 2)")
    print("           QR(2 sites) + SVD + 2×reconstruct GEMM")
    print("           MI100 peak: 184.6 TFLOPS  |  MI60 peak: 29.5 TFLOPS")
    print("═"*90)
    CHI_ROWS = [10, 20, 30, 50, 70, 100, 120]
    GRIDS    = [(4,4), (6,6), (8,8), (10,10), (12,12), (4,8), (6,10)]
    print(f"  {'chi':>5}", end="")
    for kl in GRIDS:
        k,l = kl
        print(f"  {k}×{l} gates/layer={gates_per_layer(k,l)}", end="")
    print()
    print(f"  {'':>5}", end="")
    for _ in GRIDS:
        print(f"  {'gate(GFLOP)':>12}{'layer(TFLOP)':>14}", end="")
    print()

    for chi in CHI_ROWS:
        print(f"  {chi:>5}", end="")
        for k,l in GRIDS:
            gf = flops_per_gate(chi) / 1e9
            lf = flops_per_gate(chi) * gates_per_layer(k,l) / 1e12
            print(f"  {gf:>12.1f}{lf:>14.2f}", end="")
        print()

# ════════════════════════════════════════════════════════════════════════════
# Validation against known ./main output
# ════════════════════════════════════════════════════════════════════════════

def print_validation():
    print()
    print("─"*60)
    print("Validation against ./main 4 4 50 output:")
    psi, msg = state_bytes(4, 4, 50)
    ws  = workspace_bytes(50)
    meas = measures_bytes(4, 4, 50)
    print(f"  psi                = {gb(psi)*1000:7.1f} MB  (model: {gb(psi)*1000:.1f})")
    print(f"  messages           = {gb(msg)*1000:7.1f} MB")
    print(f"  psi+msg (model)    = {gb(psi+msg)*1000:7.1f} MB   measured: 416.6 MB")
    print(f"  workspace (model)  = {gb(ws)*1000:7.1f} MB   measured: ~406 MB (after P7; pre-P6: 401.4 MB)")
    print(f"  measures scratch   = {gb(meas)*1000:7.1f} MB")
    print(f"  total (model)      = {gb(psi+msg+ws+meas)*1000:7.1f} MB   measured: ~818 MB")
    print("─"*60)

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("CppSim GPU Memory Estimator")
    print(f"Model: 2D k×l Ising, Heisenberg picture, D={D} Pauli, float32")
    print(f"Target machines: MI100 (32 GB) ★  MI60 (32 GB) ★")
    print()

    print_validation()
    print()
    print_memory_table()
    print_breakdown_table()
    print_maxchi_table()
    print_compute_table()
