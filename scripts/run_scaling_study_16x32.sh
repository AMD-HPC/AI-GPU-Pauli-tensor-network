#!/bin/bash
# run_scaling_study_16x32.sh — Systematic broken-bond scaling study for 16x32.
#
# Trial plan:
#   32 trials × 1 broken bond
#   16 trials × 2 broken bonds
#    8 trials × 4 broken bonds
#    4 trials × 8 broken bonds
#    2 trials × 16 broken bonds
#    1 trial  × 32 broken bonds
#   = 63 trials total
#
# Each trial: seismic (base+pert 2-GPU parallel, chi=40, 3 steps) + δlnT detection
#
# Usage:
#   bash scripts/run_scaling_study_16x32.sh [seed] [dJ]
#   bash scripts/run_scaling_study_16x32.sh 42 -0.20

SEED=${1:-42}
DJ=${2:--0.20}
J_PERT=$(python3 -c "print(round(1.0 + $DJ, 3))")

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
CHI=40
STEPS=3
CUTOFF=1e-3
CENTERS="5,13,21,29,34,42,50,58,71,79,87,95,100,108,116,124,129,137,145,153,166,174,182,190,195,203,211,219,261,269,277,285,290,298,306,314,327,335,343,351,356,364,372,380,385,393,401,409,422,430,438,446,451,459,467,475"
LOGBASE="logs/scaling_16x32"
mkdir -p $LOGBASE

echo "================================================================"
echo "=== 16x32 Scaling Study  seed=$SEED  dJ=$DJ  J_pert=$J_PERT ==="
echo "================================================================"
echo "Plan: 32×1 + 16×2 + 8×4 + 4×8 + 2×16 + 1×32 = 63 trials"
echo ""

TRIAL=0
RESULTS=""

run_trial() {
    local N_BONDS=$1
    local TRIAL_ID=$2
    local TRIAL_SEED=$((SEED + TRIAL_ID * 100 + N_BONDS))

    LOGDIR="$LOGBASE/nb${N_BONDS}/trial_${TRIAL_ID}"
    mkdir -p $LOGDIR
    BROKEN_FILE="$LOGDIR/broken.json"

    # Break N_BONDS random bonds
    python3 scripts/breaker.py \
        --graph $GRAPH \
        --bond $(python3 -c "
import random, json
random.seed($TRIAL_SEED)
g = json.load(open('$GRAPH'))
bonds = random.sample(range(len(g['edges'])), $N_BONDS)
print(','.join(map(str, bonds)))
") \
        --coupling $DJ \
        --out $BROKEN_FILE > /dev/null 2>&1

    # Read broken bond list and build perturb arg
    BOND_TAG=$(python3 -c "
import json
br = json.load(open('$BROKEN_FILE'))
bonds = br.get('bonds', [br.get('bond')])
print(';'.join(map(str, bonds)))
")
    PERTURB="${BOND_TAG},${J_PERT}"

    # Run base + pert in parallel on GPU 0 and GPU 1
    ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL \
        > $LOGDIR/seismic_base.log 2>&1 &
    PID_BASE=$!

    ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL \
        --perturb-bond=$PERTURB \
        > $LOGDIR/seismic_pert.log 2>&1 &
    PID_PERT=$!

    wait $PID_BASE $PID_PERT

    # Detect
    RESULT=$(python3 scripts/detect_delta_logT.py \
        --logdir $LOGDIR \
        --graph $GRAPH \
        --broken $BROKEN_FILE \
        --top_k $N_BONDS \
        --thr 0.01 2>/dev/null | grep "^Result:")

    echo "nb=${N_BONDS} trial=${TRIAL_ID}  ${RESULT}"
    echo "$N_BONDS $TRIAL_ID $RESULT" >> $LOGBASE/summary.txt
}

# 32 × 1-bond trials
echo "--- 1-bond trials (32) ---"
for i in $(seq 1 32); do run_trial 1 $i; done

# 16 × 2-bond trials
echo "--- 2-bond trials (16) ---"
for i in $(seq 1 16); do run_trial 2 $i; done

# 8 × 4-bond trials
echo "--- 4-bond trials (8) ---"
for i in $(seq 1 8); do run_trial 4 $i; done

# 4 × 8-bond trials
echo "--- 8-bond trials (4) ---"
for i in $(seq 1 4); do run_trial 8 $i; done

# 2 × 16-bond trials
echo "--- 16-bond trials (2) ---"
for i in $(seq 1 2); do run_trial 16 $i; done

# 1 × 32-bond trial
echo "--- 32-bond trial (1) ---"
run_trial 32 1

echo ""
echo "=== Summary ==="
cat $LOGBASE/summary.txt
