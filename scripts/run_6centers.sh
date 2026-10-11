#!/bin/bash
# run_6centers.sh — Run all 6 individual center experiments in parallel
#
# Each center gives a directional signal toward the broken bond.
# The intersection of 6 directions triangulates the bond location.
#
# Usage:
#   bash scripts/run_6centers.sh [bond] [delta_J]
#   bash scripts/run_6centers.sh 12 0.20
#
# Results: logs/tomo_6c/
# Summary: logs/tomo_6c/summary.txt

BOND=${1:-12}           # single (12) or multiple (12;13) — semicolon-separated
DJ=${2:-0.20}
J_NEW=$(python3 -c "print(f'{1.0 + $DJ:.3f}')")
BOND_TAG="${BOND//;/_}"   # filename-safe
BOND_PY="${BOND//;/,}"    # Python breaker.py uses commas
STEPS=7
CHI=50

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

LOGDIR="logs/tomo_7c_7step"
mkdir -p $LOGDIR

GRAPH="data/graphs/rect_7x7.json"
BINARY="./main_pauli_sweep_rect7x7"
CUTOFF=1e-3
CENTERS=(25 10 34 30 6 32)

echo "=== 6-center individual experiment ===" | tee $LOGDIR/summary.txt
echo "Bond=$BOND  dJ=$DJ  J_new=$J_NEW  chi=$CHI  steps=$STEPS" | tee -a $LOGDIR/summary.txt
echo "$(date)" | tee -a $LOGDIR/summary.txt
echo "" | tee -a $LOGDIR/summary.txt

# C_breaker config
python3 scripts/breaker.py \
    --graph $GRAPH --bond "$BOND_PY" --coupling $DJ \
    --out $LOGDIR/broken.json > /dev/null 2>&1

# ── Run one center ─────────────────────────────────────────────────────────────
run_center() {
    local C=$1
    local GPU=$2
    cd /scratch/MPF/CppSim

    echo "  [GPU$GPU] Center $C starting..."

    # Sequential: baseline then perturbed on same GPU
    # (two simultaneous runs would OOM on 32GB GPU at chi=50)
    # Optimization: with 6 GPUs, baseline+perturbed could run in parallel.
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C --state=NL \
        > $LOGDIR/base_c${C}.log 2>&1 || { echo "  [GPU$GPU] Center $C baseline FAILED"; return; }

    local PERTURB_ARG="--perturb-bond=${BOND},${J_NEW}"
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C --state=NL "$PERTURB_ARG" \
        > $LOGDIR/pert_c${C}.log 2>&1 || { echo "  [GPU$GPU] Center $C perturbed FAILED"; return; }

    RESULT=$(cd /scratch/MPF/CppSim && python3 scripts/searcher.py \
        --baseline  $LOGDIR/base_c${C}.log \
        --perturbed $LOGDIR/pert_c${C}.log \
        --graph $GRAPH --centers $C \
        --broken $LOGDIR/broken.json \
        --out $LOGDIR/result_c${C}.json 2>&1) || true

    CORRECT=$(echo "$RESULT" | grep "Result:" | head -1)
    DECLARES=$(echo "$RESULT" | grep "declares:" | head -1)
    MAX_L=$(echo "$RESULT" | grep "max L (red)" | head -1)
    MIN_L=$(echo "$RESULT" | grep "min L (blue)" | head -1)
    TRIGGER=$(echo "$RESULT" | grep "Trigger sites" | head -1)

    echo "  [GPU$GPU] Center $C: $CORRECT"
    echo "  $TRIGGER"
    echo "  $MAX_L"
    echo "  $MIN_L"
    echo "center=$C | $CORRECT | $DECLARES" >> $LOGDIR/summary.txt

    # Generate per-center cross-entropy figure
    python3 scripts/make_ce_gif.py \
        --baseline  $LOGDIR/base_c${C}.log \
        --perturbed $LOGDIR/pert_c${C}.log \
        --graph $GRAPH \
        --broken $LOGDIR/broken.json \
        --centers $C \
        --out $LOGDIR/ce_c${C}.gif \
        --fps 2 --size 5 > /dev/null 2>&1 || true

    echo "  [GPU$GPU] Center $C done."
}

# ── 3 GPUs, 2 centers each, sequential within GPU ─────────────────────────────
echo "Launching 6 centers on 3 GPUs..."

( run_center ${CENTERS[0]} 0; run_center ${CENTERS[1]} 0 ) &
( run_center ${CENTERS[2]} 1; run_center ${CENTERS[3]} 1 ) &
( run_center ${CENTERS[4]} 2; run_center ${CENTERS[5]} 2 ) &
wait

echo "" | tee -a $LOGDIR/summary.txt
echo "=== Summary ===" | tee -a $LOGDIR/summary.txt
cat $LOGDIR/summary.txt

# ── Combined search using all 6 baselines/perturbeds ──────────────────────────
echo ""
echo "=== Combined search (all 6 centers) ==="
python3 scripts/searcher.py \
    --baseline  $LOGDIR/base_c${CENTERS[0]}.log \
    --perturbed $LOGDIR/pert_c${CENTERS[0]}.log \
    --graph $GRAPH \
    --centers $(IFS=,; echo "${CENTERS[*]}") \
    --broken $LOGDIR/broken.json \
    --out $LOGDIR/result_combined.json 2>&1 | grep -E "declares|Result|max L|min L|Trigger"

# ── Declaration voting: collect all declared bonds across centers ──────────────
echo ""
echo "=== Declaration Voting ==="
python3 -c "
import json, glob, sys
sys.path.insert(0,'scripts')
from searcher import load_graph, build_adj, get_bond_sites

results = sorted(glob.glob('$LOGDIR/result_c*.json'))
declarations = {}  # bond_idx -> count
with open('$LOGDIR/broken.json') as f: broken = json.load(f)
true_bonds = set(broken.get('bonds', [broken.get('bond',-1)]))
g = load_graph('$GRAPH')

for rf in results:
    with open(rf) as f: r = json.load(f)
    b = r.get('predicted_bond', -1)
    if b >= 0:
        declarations[b] = declarations.get(b, 0) + 1

print('  Bond declarations:')
for b, count in sorted(declarations.items(), key=lambda x: -x[1]):
    a, sb = get_bond_sites(g['edges'][b])
    tag = '← TRUE' if b in true_bonds else ''
    print(f'    Bond {b:3d} ({a}↔{sb}): {count} vote(s) {tag}')

declared_set = set(declarations.keys())
found = declared_set & true_bonds
print(f'  True bonds found by voting: {sorted(found)} / {sorted(true_bonds)}')
correct = (found == true_bonds)
print(f'  Voting result: {\"✓ ALL CORRECT\" if correct else \"✗ PARTIAL\" if found else \"✗ NONE FOUND\"}')
" 2>/dev/null

echo ""
echo "=== DONE $(date) ==="
echo "Results: $LOGDIR/"
echo "GIFs:    $LOGDIR/ce_c*.gif"
