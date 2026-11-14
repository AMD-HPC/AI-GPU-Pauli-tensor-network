#!/bin/bash
# run_seismic_checkerboard_16x32.sh — 2-scan staggered checkerboard for 16x32.
#
# 2 scans × 128 centers, max dist=2 hops, guaranteed sign flip per bond.
# Every bond has one endpoint in scan 1, one in scan 2.
# Base + pert parallel on GPU 0+1.
#
# Usage:
#   bash scripts/run_seismic_checkerboard_16x32.sh <bond> <J_pert> [trial_id]

BOND=${1:-200}
J_PERT=${2:-0.800}
TRIAL=${3:-1}

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
PLAN="data/scans/rect_16x32_checkerboard.json"
CHI=40; STEPS=3; CUTOFF=1e-3
LOGDIR="logs/checkerboard_16x32/trial_${TRIAL}"
mkdir -p $LOGDIR

SCAN1=$(python3 -c "
import json
p = json.load(open('data/scans/rect_16x32_checkerboard.json'))
print(','.join(map(str, p['placements'][0]['centers'])))")

SCAN2=$(python3 -c "
import json
p = json.load(open('data/scans/rect_16x32_checkerboard.json'))
print(','.join(map(str, p['placements'][1]['centers'])))")

PERTURB="${BOND},${J_PERT}"

echo "================================================================"
echo "=== Checkerboard 2-Scan 16x32  Trial=$TRIAL  Bond=$BOND  J=$J_PERT ==="
echo "================================================================"
echo "Chi=$CHI  Steps=$STEPS  Scans: 128+128 centers  max_dist=2 hops"
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
    grep "^STEP\|Workspace\|OOM" $LOGDIR/scan${SID}_base.log | tail -2
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
