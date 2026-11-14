#!/bin/bash
# run_seismic_period6_chi_16x32.sh — Period-6 seismic with configurable chi and v_hop.
#
# Usage:
#   bash scripts/run_seismic_period6_chi_16x32.sh <bond> <J_pert> [trial_id] [chi] [v_hop]
#   bash scripts/run_seismic_period6_chi_16x32.sh 200 0.800 1 60 0.9

BOND=${1:-200}
J_PERT=${2:-0.800}
TRIAL=${3:-1}
CHI=${4:-60}
VHOP=${5:-0.9}

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
PLAN="data/scans/rect_16x32_period6.json"
STEPS=3; CUTOFF=1e-3
LOGDIR="logs/period6_chi${CHI}_16x32/trial_${TRIAL}"
mkdir -p $LOGDIR

SCAN1="1,7,13,19,25,31,193,199,205,211,217,223,385,391,397,403,409,415"
SCAN2="100,106,112,118,124,292,298,304,310,316,484,490,496,502,508"
PERTURB="${BOND},${J_PERT}"

echo "================================================================"
echo "=== Period-6 chi=${CHI} v_hop=${VHOP}  Trial=${TRIAL}  Bond=${BOND} ==="
echo "================================================================"

run_scan() {
    local SID=$1; local CENTERS=$2
    echo "--- Scan $SID ---"
    ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL --v_hop=$VHOP \
        > $LOGDIR/scan${SID}_base.log 2>&1 &
    PID_BASE=$!
    ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTERS --state=NL --v_hop=$VHOP \
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

echo "Done. Logs: $LOGDIR"
