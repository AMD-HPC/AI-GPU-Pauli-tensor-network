#!/bin/bash
# run_tomography_v2.sh — Hamiltonian Tomography: one experiment at a time
#
# Usage:
#   bash scripts/run_tomography_v2.sh <mode> <bond> <delta_J> [gpu]
#
#   mode: A_center<N>  — single center N (e.g. A_center10)
#         B            — multi-center chi=80 v_hop=0.5
#         C            — multi-center chi=80 no-v_hop
#
#   Examples:
#     bash scripts/run_tomography_v2.sh A_center10 12 0.20 0
#     bash scripts/run_tomography_v2.sh B 12 0.20 0
#     bash scripts/run_tomography_v2.sh C 12 0.20 0

MODE=${1:-A_center10}
BOND=${2:-12}           # single bond (12) or multiple (12;13) — semicolon-separated
DJ=${3:-0.20}
GPU=${4:-0}
J_NEW=$(python3 -c "print(f'{1.0 + $DJ:.3f}')")
# Tag-safe bond name (replace semicolons with underscores for filenames)
BOND_TAG="${BOND//;/_}"
# breaker.py uses commas; convert semicolons to commas for Python
BOND_PY="${BOND//;/,}"

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

LOGDIR="logs/tomo_v2"
mkdir -p $LOGDIR

GRAPH="data/graphs/rect_7x7.json"
BINARY="./main_pauli_sweep_rect7x7"
CUTOFF=1e-3
CENTERS_STR="25,10,34,30,6,32"

echo "=== Tomography v2: mode=$MODE bond=$BOND dJ=$DJ J_new=$J_NEW GPU=$GPU ==="

# Generate breaker config (use comma-separated for Python)
python3 scripts/breaker.py \
    --graph $GRAPH \
    --bond "$BOND_PY" --coupling $DJ \
    --out $LOGDIR/broken_bond${BOND_TAG}.json > /dev/null 2>&1

BROKEN_JSON=$LOGDIR/broken_bond${BOND_TAG}.json

# ── Determine parameters ───────────────────────────────────────────────────────
if [[ $MODE == A_center* ]]; then
    CENTER=${MODE#A_center}
    CHI=50
    STEPS=3
    CENTER_ARG="--center=$CENTER"
    VHOP_ARG="--v_hop=0.5"
    TAG="${MODE}_bond${BOND_TAG}"
elif [[ $MODE == "B" ]]; then
    CENTER_ARG="--center=$CENTERS_STR"
    CHI=80
    STEPS=5
    VHOP_ARG=""   # no v_hop — all bonds must be reachable
    TAG="B_bond${BOND_TAG}"
elif [[ $MODE == "C" ]]; then
    CENTER_ARG="--center=$CENTERS_STR"
    CHI=80
    STEPS=3
    VHOP_ARG=""
    TAG="C_bond${BOND_TAG}"
else
    echo "Unknown mode: $MODE"
    echo "Use: A_center<N>, B, or C"
    exit 1
fi

BASE_LOG=$LOGDIR/base_${TAG}.log
PERT_LOG=$LOGDIR/pert_${TAG}.log
RESULT_JSON=$LOGDIR/result_${TAG}.json
CE_PNG=$LOGDIR/ce_${TAG}.png

# ── Baseline + Perturbed in parallel on GPU and GPU+1 ─────────────────────────
GPU2=$(( GPU + 1 ))
echo ""
echo "[1-2/3] Baseline (GPU$GPU) + Perturbed (GPU$GPU2) in parallel"

ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
    $CENTER_ARG --state=NL $VHOP_ARG \
    > $BASE_LOG 2>&1 &
PID_BASE=$!

ROCR_VISIBLE_DEVICES=$GPU2 $BINARY $CHI $STEPS $CUTOFF \
    $CENTER_ARG --state=NL $VHOP_ARG \
    "--perturb-bond=${BOND},${J_NEW}" \
    > $PERT_LOG 2>&1 &
PID_PERT=$!

wait $PID_BASE $PID_PERT
echo "  Baseline done.  $(grep '^STEP' $BASE_LOG | tail -1 | cut -c1-60)"
echo "  Perturbed done. $(grep 'perturb-bonds\|PTM_elements' $PERT_LOG | head -1)"

# ── Search ────────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Search..."
if [[ $MODE == A_center* ]]; then
    SEARCH_CENTERS=$CENTER
else
    SEARCH_CENTERS=$CENTERS_STR
fi

python3 scripts/searcher.py \
    --baseline  $BASE_LOG \
    --perturbed $PERT_LOG \
    --graph $GRAPH \
    --centers $SEARCH_CENTERS \
    --broken $BROKEN_JSON \
    --out $RESULT_JSON

# ── Cross-entropy figure ──────────────────────────────────────────────────────
echo ""
echo "Generating figure: $CE_PNG"
python3 -c "
import sys; sys.path.insert(0,'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
import numpy as np, matplotlib, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

nc_b, ns, data_base = parse_pauli_log('$BASE_LOG', 'NL')
nc_p, ns, data_pert = parse_pauli_log('$PERT_LOG', 'NL')
g = load_graph('$GRAPH')
adj, bond_map, all_sites = build_adj(g)
pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
xs = [pos[s][0] for s in sorted(all_sites)]
ys = [pos[s][1] for s in sorted(all_sites)]

with open('$BROKEN_JSON') as f: broken = json.load(f)
sa, sb = broken['site_a'], broken['site_b']

nc = min(nc_b, nc_p)
fig, axes = plt.subplots(1, nc, figsize=(5*nc, 5))
if nc == 1: axes = [axes]
fig.suptitle('$MODE bond=$BOND (${sa}↔${sb}) dJ=$DJ chi=$CHI', fontsize=11)
FLOOR=1e-6
for i, t in enumerate(range(nc)):
    ax = axes[i]
    ce = np.array([-np.log(max(data_pert[t,vi],FLOOR)/max(data_base[t,vi],FLOOR))
                   for vi in range(ns)])
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    sc = ax.scatter(xs, ys, c=ce, cmap='RdBu_r', norm=norm, s=400, zorder=3)
    for s in [sa, sb]:
        ax.scatter([pos[s][0]], [pos[s][1]], s=600,
            facecolors='none', edgecolors='gold', linewidths=3, zorder=5)
    for c in [25,10,34,30,6,32]:
        ax.scatter([pos[c][0]], [pos[c][1]], marker='*', s=300, color='white', zorder=6)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_title(f'Cycle {t+1}', fontsize=10)
    ax.set_aspect('equal'); ax.axis('off')
plt.tight_layout()
plt.savefig('$CE_PNG', dpi=100, bbox_inches='tight')
print('Saved:', '$CE_PNG')
" 2>/dev/null

echo ""
echo "=== DONE ==="
echo "Logs: $BASE_LOG  $PERT_LOG"
echo "Result: $RESULT_JSON"
echo "Figure: $CE_PNG"
