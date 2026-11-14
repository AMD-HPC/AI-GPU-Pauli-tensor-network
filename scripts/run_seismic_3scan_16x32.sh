#!/bin/bash
# run_seismic_3scan_16x32.sh — 3-scan seismic tomography for 16x32 (512-qubit).
#
# Replicates the 7x7 3-scan strategy (corners+center, quadrant mid, edge mid)
# scaled to 8x8 tiles tiled 2x4 across 16x32.
#
# 3 scans x 2 runs (base+pert parallel on GPU 0+1) = 6 GPU runs total.
# Each scan: chi=40, STEPS=3, no v_hop.
# Detection: BFS + δlnT aggregated across all 3 scan logs.
#
# Usage:
#   bash scripts/run_seismic_3scan_16x32.sh <bond> <J_pert> [trial_id]
#   bash scripts/run_seismic_3scan_16x32.sh 200 0.800 1

BOND=${1:-200}
J_PERT=${2:-0.800}
TRIAL=${3:-1}

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
PLAN="data/scans/rect_16x32_3scan.json"
CHI=40
STEPS=3
CUTOFF=1e-3
LOGDIR="logs/seismic3_16x32/trial_${TRIAL}"
mkdir -p $LOGDIR

SCAN1="1,8,9,16,17,24,25,32,101,109,117,125,225,232,233,240,241,248,249,256,257,264,265,272,273,280,281,288,357,365,373,381,481,488,489,496,497,504,505,512"
SCAN2="35,39,43,47,51,55,59,63,195,199,203,207,211,215,219,223,291,295,299,303,307,311,315,319,451,455,459,463,467,471,475,479"
SCAN3="5,13,21,29,129,136,137,144,145,152,153,160,229,237,245,253,261,269,277,285,385,392,393,400,401,408,409,416,485,493,501,509"

PERTURB="${BOND},${J_PERT}"

echo "================================================================"
echo "=== 3-Scan Seismic 16x32  Trial=$TRIAL  Bond=$BOND  J=$J_PERT ==="
echo "================================================================"
echo "Chi=$CHI  Steps=$STEPS  Scans: 40+32+32 centers"
echo ""

run_scan() {
    local SID=$1
    local CENTERS=$2
    echo "--- Scan $SID ($(echo $CENTERS | tr ',' '\n' | wc -l) centers) ---"
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
    --logdir $LOGDIR \
    --graph $GRAPH \
    --broken logs/mawgm_16x32/trial_${TRIAL}/broken.json \
    --top_k 1 --thr 0.01 2>/dev/null || \
python3 scripts/detect_delta_logT.py \
    --logdir $LOGDIR \
    --graph $GRAPH \
    --top_k 1 --thr 0.01

echo ""
echo "--- Detection: BFS ---"
python3 scripts/bfs_detect.py \
    --logdir $LOGDIR \
    --plan $PLAN \
    --graph $GRAPH \
    --broken logs/mawgm_16x32/trial_${TRIAL}/broken.json \
    --top_k 1 --thr 1e-4 2>/dev/null || \
python3 scripts/bfs_detect.py \
    --logdir $LOGDIR \
    --plan $PLAN \
    --graph $GRAPH \
    --top_k 1 --thr 1e-4

echo ""
echo "Done. Logs: $LOGDIR"
