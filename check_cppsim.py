"""
check_cppsim.py — per-layer physics observables for CppSim logs.

Parses per layer:
  C(t)          autocorrelation
  sweeps        BP sweep count
  delta         BP convergence delta
  gate_err      max truncation weight fraction (new column)
  Se/logX       max bond entanglement saturation S_e/log(chi_max) (new column)
  gap           min singular value gap at truncation cut (new column)

Usage:
  python check_cppsim.py                  # all *.log in logs/
  python check_cppsim.py file1.log ...
"""

import re, sys, glob, os

# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def parse_log(path):
    re_grid    = re.compile(r"Grid:\s*(\d+)x(\d+)\s+chi_max=(\d+)")
    re_steps   = re.compile(r"Steps:\s*(\d+)")
    re_layer   = re.compile(
        r"^(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+"   # layer chi_min chi_max C(t)
        r"([\d.eE+\-]+)\s+"                         # total(s)
        r"([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+"  # trotter bp Cfn
        r"(\d+)\s+([\d.eE+\-]+)"                    # sweeps delta
        r"(?:\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+))?"  # gate_err Se/logX gap (optional)
    )

    meta   = dict(nx=None, ny=None, chi_max=None, n_steps=None)
    layers = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if m := re_grid.search(line):
                meta["nx"]      = int(m.group(1))
                meta["ny"]      = int(m.group(2))
                meta["chi_max"] = int(m.group(3))
                continue
            if m := re_steps.search(line):
                meta["n_steps"] = int(m.group(1))
                continue
            if m := re_layer.match(line.strip()):
                layers.append(dict(
                    layer    = int(m.group(1)),
                    chi_min  = int(m.group(2)),
                    chi_max  = int(m.group(3)),
                    C        = float(m.group(4)),
                    total_s  = float(m.group(5)),
                    trotter  = float(m.group(6)),
                    bp       = float(m.group(7)),
                    cfn      = float(m.group(8)),
                    sweeps   = int(m.group(9)),
                    delta    = float(m.group(10)),
                    gate_err = float(m.group(11)) if m.group(11) else None,
                    Se_logX  = float(m.group(12)) if m.group(12) else None,
                    gap      = float(m.group(13)) if m.group(13) else None,
                ))
    return meta, layers

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _f(v, w, d, sci=False):
    if v is None: return "—".center(w)
    return (f"{v:.{d}e}" if sci else f"{v:.{d}f}").rjust(w)

FLAG_SWEEPS   = 50     # BP struggling
FLAG_DELTA    = 1e-3   # BP not converged
FLAG_SE       = 0.8    # bond at capacity
FLAG_GAP      = 0.1    # ambiguous cut
FLAG_ERR      = 1e-4   # non-trivial truncation

def print_table(path, meta, layers):
    label = os.path.basename(path)
    grid  = f"{meta['nx']}×{meta['ny']}" if meta["nx"] else "?×?"
    chi   = meta["chi_max"] or "?"
    steps = meta["n_steps"] or "?"
    print(f"\n{grid}  chi={chi}  steps={steps}  —  {label}")

    has_svd = any(r["gate_err"] is not None for r in layers)

    hdr = (f"{'L':>3} {'χmin':>5} {'χmax':>5} {'C(t)':>10}  "
           f"{'sweeps':>6} {'delta':>10}  ")
    if has_svd:
        hdr += f"{'gate_err':>10} {'Se/logX':>8} {'gap':>7}"
    sep = "─" * len(hdr)
    print(sep); print(hdr); print(sep)

    for r in layers:
        f_sw  = "!" if r["sweeps"] >= FLAG_SWEEPS else " "
        f_dl  = "!" if r["delta"]  >= FLAG_DELTA  else " "
        f_se  = ("!" if r["Se_logX"]  is not None and r["Se_logX"]  >= FLAG_SE  else " ") if has_svd else ""
        f_ga  = ("!" if r["gap"]      is not None and r["gap"]      <= FLAG_GAP else " ") if has_svd else ""
        f_er  = ("!" if r["gate_err"] is not None and r["gate_err"] >= FLAG_ERR  else " ") if has_svd else ""

        row = (f"{r['layer']:>3} "
               f"{r['chi_min']:>5} "
               f"{r['chi_max']:>5} "
               f"{r['C']:>10.6f}  "
               f"{r['sweeps']:>6}{f_sw}"
               f"{_f(r['delta'],10,2,sci=True)}{f_dl}  ")
        if has_svd:
            row += (f"{_f(r['gate_err'],10,2,sci=True)}{f_er}"
                    f"{_f(r['Se_logX'],8,4)}{f_se}"
                    f"{_f(r['gap'],7,4)}{f_ga}")
        print(row)

    print(sep)
    flags = "  ! flags: sweeps≥50  delta≥1e-3"
    if has_svd:
        flags += "  gate_err≥1e-4  Se/logX≥0.80  gap≤0.10"
    print(flags)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "*.log")

def main():
    paths = sys.argv[1:] or sorted(glob.glob(DEFAULT))
    if not paths:
        print(f"No logs found matching: {DEFAULT}"); sys.exit(1)
    for path in paths:
        if not os.path.isfile(path):
            print(f"Not found: {path}"); continue
        meta, layers = parse_log(path)
        if layers:
            print_table(path, meta, layers)
    print()

if __name__ == "__main__":
    main()
