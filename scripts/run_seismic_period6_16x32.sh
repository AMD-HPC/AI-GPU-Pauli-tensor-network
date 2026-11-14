#!/bin/bash
# run_seismic_period6_16x32.sh — 2-scan period-6 seismic for 16x32.
#
# 2 scans × 18+15=33 centers total. Opposing wavefronts meet in 2 steps.
# Period 6 spacing — sparse enough for v_hop to work.
#
# Usage:
#   bash scripts/run_seismic_period6_16x32.sh <bond> <J_pert> [trial_id]

BOND=${1:-200}
J_PERT=${2:-0.800}
TRIAL=${3:-1}

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
PLAN="data/scans/rect_16x32_period6.json"
CHI=40; STEPS=3; CUTOFF=1e-3; VHOP="--v_hop=0.9"
LOGDIR="logs/period6_vhop09_16x32/trial_${TRIAL}"
mkdir -p $LOGDIR

SCAN1="1,7,13,19,25,31,193,199,205,211,217,223,385,391,397,403,409,415"
SCAN2="100,106,112,118,124,292,298,304,310,316,484,490,496,502,508"

PERTURB="${BOND},${J_PERT}"

echo "================================================================"
echo "=== Period-6 2-Scan 16x32  Trial=$TRIAL  Bond=$BOND  J=$J_PERT ==="
echo "================================================================"
echo "Chi=$CHI  Steps=$STEPS  Scans: 18+15=33 centers  union_max=4 hops"
echo ""

run_scan() {
    local SID=$1; local CENTERS=$2
    echo "--- Scan $SID ---"
    ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL $VHOP \
        > $LOGDIR/scan${SID}_base.log 2>&1 &
    PID_BASE=$!
    ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL $VHOP \
        --perturb-bond=$PERTURB \
        > $LOGDIR/scan${SID}_pert.log 2>&1 &
    PID_PERT=$!
    wait $PID_BASE $PID_PERT
    grep "^STEP" $LOGDIR/scan${SID}_base.log | tail -1
}

run_scan 1 "$SCAN1"
run_scan 2 "$SCAN2"

echo ""
echo "--- Detection: δlnT ---"
python3 scripts/detect_delta_logT.py \
    --logdir $LOGDIR --graph $GRAPH \
    --broken logs/mawgm_16x32/trial_${TRIAL}/broken.json \
    --top_k 1 --thr 0.01

echo ""
echo "--- Detection: BFS ---"
python3 scripts/bfs_detect.py \
    --logdir $LOGDIR --plan $PLAN --graph $GRAPH \
    --broken logs/mawgm_16x32/trial_${TRIAL}/broken.json \
    --top_k 1 --thr 1e-4

echo "Done. Logs: $LOGDIR"
