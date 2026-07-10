"""
plot_ct_convergence.py — C(t) and chi convergence across multiple chi values

Reads logs/chi_convergence/chi_{N}.log files and produces:
  1. C(t) vs cycle for all chi values (convergence check)
  2. max_chi vs cycle (entanglement growth)
  3. Total time vs chi (performance scaling)
  4. C(t) error vs chi relative to largest chi (accuracy vs cost)

Usage:
  python3 plot_ct_convergence.py [log_dir]
  python3 plot_ct_convergence.py logs/chi_convergence
"""

import sys
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/chi_convergence"

# ── Parse log files ───────────────────────────────────────────────────────────

def parse_log(path):
    """Parse main_heavyhex log — returns dict with chi_max, layers."""
    chi_max_param = None
    layers = []
    with open(path) as f:
        for line in f:
            m = re.search(r"chi_max=(\d+)", line)
            if m and chi_max_param is None:
                chi_max_param = int(m.group(1))
            # data line: layer chi_min chi_max C(t) total(s) ...
            m = re.match(
                r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)",
                line)
            if m and int(m.group(1)) > 0:
                layers.append({
                    "cycle":   int(m.group(1)),
                    "chi_min": int(m.group(2)),
                    "chi_max": int(m.group(3)),
                    "Ct":      float(m.group(4)),
                    "total_s": float(m.group(5)),
                })
    return {"chi_param": chi_max_param, "layers": layers}

# Load all logs
chi_values = [50, 100, 150, 200, 250, 300, 350, 400, 430]
data = {}
for chi in chi_values:
    path = os.path.join(LOG_DIR, f"chi_{chi}.log")
    if os.path.exists(path):
        d = parse_log(path)
        if d["layers"]:
            data[chi] = d
            print(f"  chi={chi}: {len(d['layers'])} cycles loaded")
    else:
        print(f"  chi={chi}: not found ({path})")

if not data:
    print("No data found. Run run_chi_convergence.sh first.")
    sys.exit(1)

chis  = sorted(data.keys())
cmap  = cm.plasma
colors = {chi: cmap(i / max(len(chis)-1, 1)) for i, chi in enumerate(chis)}

out_dir = LOG_DIR
os.makedirs(out_dir, exist_ok=True)

# ── Figure 1: C(t) vs cycle ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
for chi in chis:
    layers = data[chi]["layers"]
    cycles = [0] + [l["cycle"] for l in layers]
    Ct     = [1.0] + [l["Ct"] for l in layers]
    ax.plot(cycles, Ct, color=colors[chi], linewidth=1.8,
            label=f"χ={chi}", marker="o", markersize=3)

ax.set_xlabel("Cycle", fontsize=12)
ax.set_ylabel("C(t) = ⟨Z(t)Z(0)⟩", fontsize=12)
ax.set_title("C(t) convergence with χ — heavy_hex_3x3  center=site13", fontsize=12)
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(alpha=0.3)
ax.set_xlim(0)
out = os.path.join(out_dir, "ct_convergence.png")
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Figure 2: max_chi vs cycle ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
for chi in chis:
    layers = data[chi]["layers"]
    cycles  = [l["cycle"] for l in layers]
    chi_max = [l["chi_max"] for l in layers]
    ax.plot(cycles, chi_max, color=colors[chi], linewidth=1.8,
            label=f"cap={chi}", marker="o", markersize=3)
    # Mark saturation point
    for i, (c, cm_) in enumerate(zip(cycles, chi_max)):
        if cm_ >= chi:
            ax.axvline(x=c, color=colors[chi], linestyle=":", alpha=0.4)
            break

ax.set_xlabel("Cycle", fontsize=12)
ax.set_ylabel("Max bond dimension χ", fontsize=12)
ax.set_title("Bond dimension growth — saturation at χ cap", fontsize=12)
ax.legend(loc="lower right", fontsize=9, ncol=2)
ax.grid(alpha=0.3)
out = os.path.join(out_dir, "chi_growth.png")
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Figure 3: C(t) error vs chi (relative to largest chi) ────────────────────

ref_chi = max(chis)
ref_data = data[ref_chi]["layers"]
ref_Ct = {l["cycle"]: l["Ct"] for l in ref_data}

fig, ax = plt.subplots(figsize=(10, 5))
for chi in chis[:-1]:  # skip reference
    layers = data[chi]["layers"]
    cycles = [l["cycle"] for l in layers if l["cycle"] in ref_Ct]
    errors = [abs(l["Ct"] - ref_Ct[l["cycle"]]) for l in layers if l["cycle"] in ref_Ct]
    ax.semilogy(cycles, errors, color=colors[chi], linewidth=1.8,
                label=f"χ={chi}", marker="o", markersize=3)

ax.set_xlabel("Cycle", fontsize=12)
ax.set_ylabel(f"|C(t,χ) - C(t,χ={ref_chi})|", fontsize=12)
ax.set_title(f"C(t) truncation error vs χ (reference: χ={ref_chi})", fontsize=12)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(alpha=0.3, which="both")
out = os.path.join(out_dir, "ct_error.png")
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Figure 4: total time at last cycle vs chi ─────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
last_times = []
for chi in chis:
    layers = data[chi]["layers"]
    if layers:
        t = layers[-1]["total_s"] * len(layers)  # cumulative approx
        t_last = layers[-1]["total_s"]
        last_times.append((chi, t_last))

chi_arr = [x[0] for x in last_times]
t_arr   = [x[1] for x in last_times]
ax.semilogy(chi_arr, t_arr, "o-", color="steelblue", linewidth=2, markersize=8)
ax.set_xlabel("χ cap", fontsize=12)
ax.set_ylabel("Time for last cycle (s)", fontsize=12)
ax.set_title("Compute cost at saturation — last cycle time vs χ", fontsize=12)
ax.grid(alpha=0.3, which="both")
for chi, t in zip(chi_arr, t_arr):
    ax.annotate(f"χ={chi}\n{t:.1f}s", (chi, t),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
out = os.path.join(out_dir, "timing_vs_chi.png")
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("═" * 70)
print("  C(t) CONVERGENCE SUMMARY")
print(f"  Reference: χ={ref_chi}")
print("═" * 70)
print(f"  {'χ':>5}  {'cycles':>6}  {'C(t=10)':>10}  {'C(t=20)':>10}  "
      f"{'err@10':>10}  {'err@20':>10}  {'t_last(s)':>10}")
print("  " + "─" * 68)
for chi in chis:
    layers = data[chi]["layers"]
    ct10 = next((l["Ct"] for l in layers if l["cycle"]==10), None)
    ct20 = next((l["Ct"] for l in layers if l["cycle"]==20), None)
    tlast = layers[-1]["total_s"] if layers else 0
    ref10 = ref_Ct.get(10, None)
    ref20 = ref_Ct.get(20, None)
    e10 = f"{abs(ct10-ref10):.2e}" if ct10 and ref10 else "—"
    e20 = f"{abs(ct20-ref20):.2e}" if ct20 and ref20 else "—"
    ct10s = f"{ct10:.6f}" if ct10 else "—"
    ct20s = f"{ct20:.6f}" if ct20 else "—"
    print(f"  {chi:>5}  {len(layers):>6}  {ct10s:>10}  {ct20s:>10}  "
          f"{e10:>10}  {e20:>10}  {tlast:>10.1f}")
print("═" * 70)
