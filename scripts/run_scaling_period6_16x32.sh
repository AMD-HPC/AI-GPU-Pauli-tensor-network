#!/bin/bash
# run_scaling_period6_16x32.sh — Scaling study with period-6 2-scan seismic.
#
# 32x1 + 16x2 + 8x4 + 4x8 + 2x16 + 1x32 = 63 trials
# Period-6: 18+15=33 centers, no v_hop, chi=40, steps=3
# Base+pert parallel on GPU 0+1
#
# Usage:
#   bash scripts/run_scaling_period6_16x32.sh [seed] [dJ]

SEED=${1:-42}
DJ=${2:--0.20}
J_PERT=$(python3 -c "print(round(1.0 + $DJ, 3))")

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
PLAN="data/scans/rect_16x32_period6.json"
CHI=40; STEPS=3; CUTOFF=1e-3
SCAN1="1,7,13,19,25,31,193,199,205,211,217,223,385,391,397,403,409,415"
SCAN2="100,106,112,118,124,292,298,304,310,316,484,490,496,502,508"
LOGBASE="logs/scaling_period6_16x32"
mkdir -p $LOGBASE

echo "================================================================"
echo "=== Period-6 Scaling Study  seed=$SEED  dJ=$DJ  J_pert=$J_PERT ==="
echo "================================================================"
echo "Plan: 32x1 + 16x2 + 8x4 + 4x8 + 2x16 + 1x32 = 63 trials"
echo ""

run_trial() {
    local N_BONDS=$1
    local TRIAL_ID=$2
    local TRIAL_SEED=$((SEED + TRIAL_ID * 100 + N_BONDS))

    LOGDIR="$LOGBASE/nb${N_BONDS}/trial_${TRIAL_ID}"
    mkdir -p $LOGDIR
    BROKEN_FILE="$LOGDIR/broken.json"

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

    BOND_TAG=$(python3 -c "
import json
br = json.load(open('$BROKEN_FILE'))
bonds = br.get('bonds', [br.get('bond')])
print(';'.join(map(str, bonds)))
")
    PERTURB="${BOND_TAG},${J_PERT}"

    # Scan 1: base+pert parallel GPU 0+1
    ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
        --center=$SCAN1 --state=NL \
        > $LOGDIR/scan1_base.log 2>&1 &
    ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$SCAN1 --state=NL \
        --perturb-bond=$PERTURB \
        > $LOGDIR/scan1_pert.log 2>&1 &
    wait

    # Scan 2: base+pert parallel GPU 0+1
    ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
        --center=$SCAN2 --state=NL \
        > $LOGDIR/scan2_base.log 2>&1 &
    ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$SCAN2 --state=NL \
        --perturb-bond=$PERTURB \
        > $LOGDIR/scan2_pert.log 2>&1 &
    wait

    RESULT=$(python3 scripts/detect_delta_logT.py \
        --logdir $LOGDIR \
        --graph $GRAPH \
        --broken $BROKEN_FILE \
        --top_k $N_BONDS \
        --thr 0.01 2>/dev/null | grep "^Result:")

    echo "nb=${N_BONDS} trial=${TRIAL_ID}  ${RESULT}"
    echo "$N_BONDS $TRIAL_ID $RESULT" >> $LOGBASE/summary.txt
}

for i in $(seq 1 32); do run_trial 1 $i; done
echo "--- 2-bond trials ---"
for i in $(seq 1 16); do run_trial 2 $i; done
echo "--- 4-bond trials ---"
for i in $(seq 1 8);  do run_trial 4 $i; done
echo "--- 8-bond trials ---"
for i in $(seq 1 4);  do run_trial 8 $i; done
echo "--- 16-bond trials ---"
for i in $(seq 1 2);  do run_trial 16 $i; done
echo "--- 32-bond trial ---"
run_trial 32 1

echo ""
echo "=== Summary ==="
cat $LOGBASE/summary.txt
