#!/bin/bash
# run_seismic_period6_2gpu.sh — Period-6 quincunx seismic on 2-GPU split.
#
# Two scans, each run as base+pert on 2 GPUs simultaneously.
# Scan 1: rows 0,6,12 (GPU0: rows 0,6 = 12c; GPU1: row 12 = 6c)
# Scan 2: rows 3,9,15 (GPU0: row 3 = 5c; GPU1: rows 9,15 = 10c)
# Detection: δlnT aggregated across both scans (4 log pairs).
#
# Usage:
#   bash scripts/run_seismic_period6_2gpu.sh <bond> <J_pert> [trial_id]
#   bash scripts/run_seismic_period6_2gpu.sh 200 0.800 1

BOND=${1:-200}
J_PERT=${2:-0.800}
TRIAL=${3:-1}

cd /scratch/MPF/CppSim

BINARY="./main_pauli_sweep_2gpu_rect16x32"
GRAPH="data/graphs/rect_16x32.json"
CHI=40; STEPS=3; CUTOFF=1e-3
LOGDIR="logs/seismic_period6_2gpu/trial_${TRIAL}"
mkdir -p $LOGDIR

SCAN1="1,7,13,19,25,31,193,199,205,211,217,223,385,391,397,403,409,415"
SCAN2="100,106,112,118,124,292,298,304,310,316,484,490,496,502,508"

PERTURB="${BOND},${J_PERT}"

echo "================================================================"
echo "=== Period-6 2-GPU Seismic  Trial=$TRIAL  Bond=$BOND  J=$J_PERT ==="
echo "================================================================"
echo "Chi=$CHI  Steps=$STEPS  Scan1=18c  Scan2=15c"
echo ""

run_scan() {
    local SID=$1; local CENTERS=$2
    echo "--- Scan $SID ---"
    ROCR_VISIBLE_DEVICES=0,1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS \
        > $LOGDIR/scan${SID}_base.log 2>&1 &
    PID_BASE=$!
    # Note: 2-GPU binary uses both GPUs internally, run sequentially
    wait $PID_BASE
    ROCR_VISIBLE_DEVICES=0,1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS \
        --perturb-bond=$PERTURB \
        > $LOGDIR/scan${SID}_pert.log 2>&1
    grep "^STEP" $LOGDIR/scan${SID}_base.log | tail -1
}

run_scan 1 "$SCAN1"
run_scan 2 "$SCAN2"

echo ""
echo "--- Detection: δlnT (both scans) ---"
python3 scripts/detect_delta_logT.py \
    --logdir $LOGDIR \
    --graph $GRAPH \
    --broken logs/mawgm_16x32/trial_${TRIAL}/broken.json \
    --top_k 1 --thr 0.01

echo "Done. Logs: $LOGDIR"
