#!/bin/bash
# run_tomography_v2.sh — Hamiltonian Tomography: single vs multi-center comparison
#
# Three experiment modes for one broken bond:
#   A: 6 × single-center  chi=200 v_hop=0.5 steps=3  (exact, cheap per run)
#   B: 1 × multi-center   chi=80  v_hop=0.5 steps=3  (seismic, lightcone-aware)
#   C: 1 × multi-center   chi=80  no-v_hop  steps=3  (seismic, full grid)
#
# Usage:
#   bash scripts/run_tomography_v2.sh [bond] [delta_J]
#   bash scripts/run_tomography_v2.sh 12 0.20
#
# Results: logs/tomo_v2/
# Summary: logs/tomo_v2/summary.txt

set -e
cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

BOND=${1:-12}
DJ=${2:-0.20}
J_NEW=$(python3 -c "print(f'{1.0 + $DJ:.3f}')")

LOGDIR="logs/tomo_v2"
mkdir -p $LOGDIR

GRAPH="data/graphs/rect_7x7.json"
CENTERS=(25 10 34 30 6 32)       # 6-center geodesic cover
CENTERS_STR="25,10,34,30,6,32"
BINARY="./main_pauli_sweep_rect7x7"
CUTOFF=1e-3

echo "=== Tomography v2: single vs multi-center ===" | tee $LOGDIR/summary.txt
echo "Bond: $BOND  delta_J: $DJ  J_new: $J_NEW" | tee -a $LOGDIR/summary.txt
echo "$(date)" | tee -a $LOGDIR/summary.txt
echo "" | tee -a $LOGDIR/summary.txt

# C_breaker config
python3 scripts/breaker.py \
    --graph $GRAPH \
    --bond $BOND --coupling $DJ \
    --out $LOGDIR/broken.json > /dev/null 2>&1
echo "Broken bond: $(python3 -c "
import json; b=json.load(open('$LOGDIR/broken.json'))
print(f'bond {b[\"bond\"]} ({b[\"site_a\"]}↔{b[\"site_b\"]})')
")" | tee -a $LOGDIR/summary.txt

# ── Experiment A: 6 × single-center (chi=200, v_hop=0.5, 3 cycles) ──────────
echo "" | tee -a $LOGDIR/summary.txt
echo "=== Experiment A: single-center (chi=200, v_hop=0.5, steps=3) ===" \
    | tee -a $LOGDIR/summary.txt

run_single_center() {
    local C=$1
    local GPU=$2
    local TAG="center${C}"

    # Baseline
    ROCR_VISIBLE_DEVICES=$GPU $BINARY 200 3 $CUTOFF \
        --center=$C --state=NL --v_hop=0.5 \
        > $LOGDIR/A_base_${TAG}.log 2>&1

    # Perturbed
    ROCR_VISIBLE_DEVICES=$GPU $BINARY 200 3 $CUTOFF \
        --center=$C --state=NL --v_hop=0.5 \
        --perturb-bond=${BOND},${J_NEW} \
        > $LOGDIR/A_pert_${TAG}.log 2>&1

    # Search
    RESULT=$(python3 scripts/searcher.py \
        --baseline  $LOGDIR/A_base_${TAG}.log \
        --perturbed $LOGDIR/A_pert_${TAG}.log \
        --graph $GRAPH --centers $C \
        --broken $LOGDIR/broken.json \
        --out $LOGDIR/A_result_${TAG}.json 2>&1)

    CORRECT=$(echo "$RESULT" | grep "Result:" | head -1)
    DECLARES=$(echo "$RESULT" | grep "declares:" | head -1)
    MAX_DN=$(echo "$RESULT" | grep "max |Δn|" | head -1)
    echo "  Center $C [GPU$GPU]: $MAX_DN | $DECLARES | $CORRECT"
    echo "A center=$C | $CORRECT | $DECLARES" >> $LOGDIR/summary.txt
}

# Run 6 centers across 3 GPUs (2 per GPU) in parallel
(run_single_center ${CENTERS[0]} 0; run_single_center ${CENTERS[1]} 0) &
PID0=$!
(run_single_center ${CENTERS[2]} 1; run_single_center ${CENTERS[3]} 1) &
PID1=$!
(run_single_center ${CENTERS[4]} 2; run_single_center ${CENTERS[5]} 2) &
PID2=$!
wait $PID0 $PID1 $PID2

# Combined A search: merge all 6 single-center logs
echo ""
echo "  Combined A search (merge all 6 centers):"
# Combine: concatenate all perturbed logs, use all centers for searcher
cat $LOGDIR/A_base_center*.log > $LOGDIR/A_base_combined.log
cat $LOGDIR/A_pert_center*.log > $LOGDIR/A_pert_combined.log
RESULT=$(python3 scripts/searcher.py \
    --baseline  $LOGDIR/A_base_combined.log \
    --perturbed $LOGDIR/A_pert_combined.log \
    --graph $GRAPH --centers $CENTERS_STR \
    --broken $LOGDIR/broken.json \
    --out $LOGDIR/A_result_combined.json 2>&1)
CORRECT=$(echo "$RESULT" | grep "Result:" | head -1)
DECLARES=$(echo "$RESULT" | grep "declares:" | head -1)
echo "  Combined: $DECLARES | $CORRECT"
echo "A combined | $CORRECT | $DECLARES" | tee -a $LOGDIR/summary.txt

# ── Experiment B: multi-center, chi=80, v_hop=0.5, 3 cycles ─────────────────
echo "" | tee -a $LOGDIR/summary.txt
echo "=== Experiment B: multi-center (chi=80, v_hop=0.5, steps=3) ===" \
    | tee -a $LOGDIR/summary.txt

# Baseline B
ROCR_VISIBLE_DEVICES=0 $BINARY 80 3 $CUTOFF \
    --center=$CENTERS_STR --state=NL --v_hop=0.5 \
    > $LOGDIR/B_base.log 2>&1 &
PID_B=$!

# Experiment C: multi-center, chi=80, no v_hop ───────────────────────────────
echo "=== Experiment C: multi-center (chi=80, no-v_hop, steps=3) ===" \
    | tee -a $LOGDIR/summary.txt

# Baseline C (run in parallel with B baseline)
ROCR_VISIBLE_DEVICES=1 $BINARY 80 3 $CUTOFF \
    --center=$CENTERS_STR --state=NL \
    > $LOGDIR/C_base.log 2>&1 &
PID_C=$!

wait $PID_B $PID_C
echo "  Baselines B and C done."

# Perturbed B and C in parallel
ROCR_VISIBLE_DEVICES=0 $BINARY 80 3 $CUTOFF \
    --center=$CENTERS_STR --state=NL --v_hop=0.5 \
    --perturb-bond=${BOND},${J_NEW} \
    > $LOGDIR/B_pert.log 2>&1 &
PID_B=$!

ROCR_VISIBLE_DEVICES=1 $BINARY 80 3 $CUTOFF \
    --center=$CENTERS_STR --state=NL \
    --perturb-bond=${BOND},${J_NEW} \
    > $LOGDIR/C_pert.log 2>&1 &
PID_C=$!

wait $PID_B $PID_C

# Search B
RESULT=$(python3 scripts/searcher.py \
    --baseline  $LOGDIR/B_base.log \
    --perturbed $LOGDIR/B_pert.log \
    --graph $GRAPH --centers $CENTERS_STR \
    --broken $LOGDIR/broken.json \
    --out $LOGDIR/B_result.json 2>&1)
CORRECT_B=$(echo "$RESULT" | grep "Result:" | head -1)
DECLARES_B=$(echo "$RESULT" | grep "declares:" | head -1)
MAX_B=$(echo "$RESULT" | grep "max |Δn|" | head -1)
echo "  B: $MAX_B | $DECLARES_B | $CORRECT_B"
echo "B chi=80 v_hop=0.5 | $CORRECT_B | $DECLARES_B" | tee -a $LOGDIR/summary.txt

# Search C
RESULT=$(python3 scripts/searcher.py \
    --baseline  $LOGDIR/C_base.log \
    --perturbed $LOGDIR/C_pert.log \
    --graph $GRAPH --centers $CENTERS_STR \
    --broken $LOGDIR/broken.json \
    --out $LOGDIR/C_result.json 2>&1)
CORRECT_C=$(echo "$RESULT" | grep "Result:" | head -1)
DECLARES_C=$(echo "$RESULT" | grep "declares:" | head -1)
MAX_C=$(echo "$RESULT" | grep "max |Δn|" | head -1)
echo "  C: $MAX_C | $DECLARES_C | $CORRECT_C"
echo "C chi=80 no-v_hop | $CORRECT_C | $DECLARES_C" | tee -a $LOGDIR/summary.txt

# ── Cross-entropy visualization ───────────────────────────────────────────────
echo "" | tee -a $LOGDIR/summary.txt
echo "=== Generating figures ===" | tee -a $LOGDIR/summary.txt
for MODE in B C; do
    python3 -c "
import sys; sys.path.insert(0,'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
import numpy as np, matplotlib, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

nc_b, ns, data_base = parse_pauli_log('$LOGDIR/${MODE}_base.log', 'NL')
nc_p, ns, data_pert = parse_pauli_log('$LOGDIR/${MODE}_pert.log', 'NL')
g = load_graph('$GRAPH')
adj, bond_map, all_sites = build_adj(g)
pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
xs = [pos[s][0] for s in sorted(all_sites)]
ys = [pos[s][1] for s in sorted(all_sites)]

with open('$LOGDIR/broken.json') as f: broken = json.load(f)
sa, sb = broken['site_a'], broken['site_b']

nc = min(nc_b, nc_p, 3)
fig, axes = plt.subplots(1, nc, figsize=(5*nc, 5))
if nc == 1: axes = [axes]
fig.suptitle('$MODE: L(v,t)=-ln(n_pert/n_base) | bond=$BOND (${sa}↔${sb}) dJ=$DJ', fontsize=11)
FLOOR=1e-6
for i, t in enumerate(range(nc)):
    ax = axes[i]
    ce = np.array([-np.log(max(data_pert[t,vi],FLOOR)/max(data_base[t,vi],FLOOR))
                   for vi in range(ns)])
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    ax.scatter(xs, ys, c=ce, cmap='RdBu_r', norm=norm, s=400, zorder=3)
    for s in [sa, sb]: ax.scatter([pos[s][0]], [pos[s][1]], s=600,
        facecolors='none', edgecolors='gold', linewidths=3, zorder=5)
    for c in [25,10,34,30,6,32]: ax.scatter([pos[c][0]], [pos[c][1]],
        marker='*', s=300, color='white', zorder=6)
    ax.set_title(f'Cycle {t+1}', fontsize=10)
    ax.set_aspect('equal'); ax.axis('off')
plt.tight_layout()
plt.savefig('$LOGDIR/ce_${MODE}.png', dpi=100, bbox_inches='tight')
print('Saved: $LOGDIR/ce_${MODE}.png')
" 2>/dev/null
done

echo "" | tee -a $LOGDIR/summary.txt
echo "=== DONE $(date) ===" | tee -a $LOGDIR/summary.txt
echo "" | tee -a $LOGDIR/summary.txt
echo "Results: $LOGDIR/"
echo "Summary: $LOGDIR/summary.txt"
echo "Figures: $LOGDIR/ce_B.png  $LOGDIR/ce_C.png"
