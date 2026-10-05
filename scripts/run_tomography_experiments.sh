#!/bin/bash
# run_tomography_experiments.sh — Hamiltonian Tomography Game experiments
#
# Runs a sweep of (bond, coupling) perturbations to study:
#   1. Minimum detectable coupling change
#   2. Whether different bonds are identifiable
#   3. Whether larger perturbations show error spreading
#
# Usage:
#   bash scripts/run_tomography_experiments.sh
#
# Results: logs/tomography/
# Summary: logs/tomography/summary.txt

set -e
cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

LOGDIR="logs/tomography"
mkdir -p $LOGDIR

# 3 GPUs available — assign round-robin
GPUS=(0 1 2)
GPU_IDX=0
GRAPH="data/graphs/rect_7x7.json"
CENTERS="25,10,34,30,6,32"
CHI=50
STEPS=5
CUTOFF=1e-3

echo "=== Hamiltonian Tomography Experiments $(date) ===" | tee $LOGDIR/summary.txt
echo "Grid: rect 7x7  Centers: $CENTERS  Chi: $CHI  Steps: $STEPS" | tee -a $LOGDIR/summary.txt
echo "" | tee -a $LOGDIR/summary.txt

# ── Step 1: Baseline (run once, GPU 0) ────────────────────────────────────────
echo "[Baseline] Running..."
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep_rect7x7 $CHI $STEPS $CUTOFF \
    --center=$CENTERS --state=NL \
    > $LOGDIR/baseline_NL.log 2>&1
echo "[Baseline] Done"

# ── Experiment function — runs on specified GPU ────────────────────────────────
run_experiment() {
    local BOND=$1
    local COUPLING_DELTA=$2
    local GPU_DEV=$3
    local J_NEW=$(python3 -c "print(f'{1.0 + $COUPLING_DELTA:.3f}')")
    local TAG="bond${BOND}_dJ${COUPLING_DELTA}"

    echo ""
    echo "--- Experiment [GPU$GPU_DEV]: bond=$BOND  delta_J=$COUPLING_DELTA  J_new=$J_NEW ---"

    # C_breaker
    python3 scripts/breaker.py \
        --graph $GRAPH \
        --bond $BOND --coupling $COUPLING_DELTA \
        --out $LOGDIR/broken_${TAG}.json > /dev/null 2>&1

    # Perturbed simulation
    ROCR_VISIBLE_DEVICES=$GPU_DEV ./main_pauli_sweep_rect7x7 $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL \
        --perturb-bond=${BOND},${J_NEW} \
        > $LOGDIR/perturbed_${TAG}.log 2>&1

    # C_searcher
    RESULT=$(python3 scripts/searcher.py \
        --baseline  $LOGDIR/baseline_NL.log \
        --perturbed $LOGDIR/perturbed_${TAG}.log \
        --graph $GRAPH \
        --centers $CENTERS \
        --broken $LOGDIR/broken_${TAG}.json \
        --out $LOGDIR/result_${TAG}.json 2>&1)

    # Extract key lines
    CORRECT=$(echo "$RESULT" | grep "Result:" | head -1)
    TRIGGER=$(echo "$RESULT" | grep "Trigger sites" | head -1)
    DECLARES=$(echo "$RESULT" | grep "declares:" | head -1)
    MAX_DN=$(echo "$RESULT" | grep "max |Δn|" | head -1)

    echo "  $MAX_DN"
    echo "  $TRIGGER"
    echo "  $DECLARES"
    echo "  $CORRECT"

    # Summary line
    echo "bond=$BOND dJ=$COUPLING_DELTA J_new=$J_NEW | $CORRECT | $DECLARES" \
        | tee -a $LOGDIR/summary.txt
}

# ── Experiments — 3 GPUs in parallel ──────────────────────────────────────────
# Each GPU runs its batch sequentially; all 3 batches run in parallel.

echo ""
echo "=== Launching experiments on 3 GPUs ===" | tee -a $LOGDIR/summary.txt

# GPU 0: Sensitivity sweep — bond 12, varying coupling
(
    echo "=== GPU0: Sensitivity sweep bond 12 ===" | tee -a $LOGDIR/summary.txt
    run_experiment 12  0.05  0
    run_experiment 12  0.10  0
    run_experiment 12  0.20  0
    run_experiment 12  0.50  0
    run_experiment 12  1.00  0
    run_experiment 12 -0.20  0
    run_experiment 12 -0.50  0
) &
PID0=$!

# GPU 1: Bond sweep — interior bonds, fixed coupling
(
    echo "=== GPU1: Bond sweep interior ===" | tee -a $LOGDIR/summary.txt
    for BOND in 0 12 20 30 40 50; do
        run_experiment $BOND 0.20 1
    done
) &
PID1=$!

# GPU 2: Bond sweep — boundary + large perturbations
(
    echo "=== GPU2: Bond sweep boundary + large dJ ===" | tee -a $LOGDIR/summary.txt
    for BOND in 60 70 83; do
        run_experiment $BOND 0.20 2
    done
    run_experiment 35 0.50 2
    run_experiment 42 0.50 2
    run_experiment  0 0.50 2
    run_experiment  5 0.50 2
) &
PID2=$!

wait $PID0 $PID1 $PID2
echo ""
echo "=== All GPU batches complete ===" | tee -a $LOGDIR/summary.txt

# ── Cross-entropy visualization for selected experiments ──────────────────────
echo ""
echo "=== Generating cross-entropy figures ===" | tee -a $LOGDIR/summary.txt
for BOND in 12 0 42; do
    for DJ in 0.20 0.50; do
        TAG="bond${BOND}_dJ${DJ}"
        J_NEW=$(python3 -c "print(f'{1.0 + $DJ:.2f}')")
        if [ -f $LOGDIR/perturbed_${TAG}.log ]; then
            python3 -c "
import sys; sys.path.insert(0,'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

nc_b, ns, data_base = parse_pauli_log('$LOGDIR/baseline_NL.log', 'NL')
nc_p, ns, data_pert = parse_pauli_log('$LOGDIR/perturbed_${TAG}.log', 'NL')
g = load_graph('$GRAPH')
adj, bond_map, all_sites = build_adj(g)
pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
xs = [pos[s][0] for s in sorted(all_sites)]
ys = [pos[s][1] for s in sorted(all_sites)]

import json
with open('$LOGDIR/broken_${TAG}.json') as f: broken = json.load(f)
sa, sb = broken['site_a'], broken['site_b']

fig, axes = plt.subplots(1,4,figsize=(20,5))
fig.suptitle('L(v,t)=-ln(n_pert/n_base) | bond=$BOND (${sa}↔${sb}) dJ=$DJ', fontsize=11)
FLOOR=1e-6
for i,t in enumerate([0,1,2,3]):
    ax=axes[i]
    ce=np.array([-np.log(max(data_pert[t,vi],FLOOR)/max(data_base[t,vi],FLOOR))
                 for vi in range(ns)])
    norm=mcolors.TwoSlopeNorm(vmin=-0.5,vcenter=0,vmax=0.5)
    ax.scatter(xs,ys,c=ce,cmap='RdBu_r',norm=norm,s=400,zorder=3)
    for s in [sa,sb]: ax.scatter([pos[s][0]],[pos[s][1]],s=600,
        facecolors='none',edgecolors='gold',linewidths=3,zorder=5)
    for c in [25,10,34,30,6,32]: ax.scatter([pos[c][0]],[pos[c][1]],
        marker='*',s=300,color='white',zorder=6)
    ax.set_title(f'Cycle {t+1}',fontsize=10); ax.set_aspect('equal'); ax.axis('off')
plt.tight_layout()
plt.savefig('$LOGDIR/ce_${TAG}.png',dpi=100,bbox_inches='tight')
print('Saved: $LOGDIR/ce_${TAG}.png')
" 2>/dev/null
        fi
    done
done

echo ""
echo "=== ALL DONE $(date) ===" | tee -a $LOGDIR/summary.txt
echo ""
echo "Results in: $LOGDIR/"
echo "Summary:    $LOGDIR/summary.txt"
echo "Figures:    $LOGDIR/ce_*.png"
