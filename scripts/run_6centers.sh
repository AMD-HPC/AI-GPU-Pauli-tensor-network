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

BOND=${1:-12}
DJ=${2:-0.20}
J_NEW=$(python3 -c "print(f'{1.0 + $DJ:.3f}')")
STEPS=5
CHI=50

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

LOGDIR="logs/tomo_6c"
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
    --graph $GRAPH --bond $BOND --coupling $DJ \
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

    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C --state=NL \
        "--perturb-bond=${BOND},${J_NEW}" \
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

echo ""
echo "=== DONE $(date) ==="
echo "Results: $LOGDIR/"
echo "GIFs:    $LOGDIR/ce_c*.gif"
