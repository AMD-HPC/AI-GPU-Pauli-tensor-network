#!/bin/bash
# run_seismic_quincunx_16x32.sh — Sparse quincunx 3-scan seismic for 16x32.
#
# 3 horizontal lines staggered (Roman cohort / quincunx):
#   Scan 1 (16 centers): rows 0,5,10,15  cols 0,8,16,24
#   Scan 2 (12 centers): rows 2,7,12     cols 4,12,20,28
#   Scan 3 (12 centers): rows 4,9,14     cols 2,10,18,26
#
# Usage:
#   bash scripts/run_seismic_quincunx_16x32.sh <bond> <J_pert> [trial_id]

BOND=${1:-200}
J_PERT=${2:-0.800}
TRIAL=${3:-1}

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
PLAN="data/scans/rect_16x32_quincunx.json"
CHI=40; STEPS=3; CUTOFF=1e-3
LOGDIR="logs/quincunx_16x32/trial_${TRIAL}"
mkdir -p $LOGDIR

SCAN1="1,9,17,25,161,169,177,185,321,329,337,345,481,489,497,505"
SCAN2="69,77,85,93,229,237,245,253,389,397,405,413"
SCAN3="35,43,51,59,195,203,211,219,355,363,371,379"

PERTURB="${BOND},${J_PERT}"

echo "================================================================"
echo "=== Quincunx 3-Scan 16x32  Trial=$TRIAL  Bond=$BOND  J=$J_PERT ==="
echo "================================================================"
echo "Chi=$CHI  Steps=$STEPS  Scans: 16+12+12=40 centers"
echo ""

run_scan() {
    local SID=$1; local CENTERS=$2
    echo "--- Scan $SID ---"
    ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL \
        > $LOGDIR/scan${SID}_base.log 2>&1 &
    PID_BASE=$!
    ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL \
        --perturb-bond=$PERTURB \
        > $LOGDIR/scan${SID}_pert.log 2>&1 &
    PID_PERT=$!
    wait $PID_BASE $PID_PERT
    grep "^STEP" $LOGDIR/scan${SID}_base.log | tail -1
}

run_scan 1 "$SCAN1"
run_scan 2 "$SCAN2"
run_scan 3 "$SCAN3"

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
