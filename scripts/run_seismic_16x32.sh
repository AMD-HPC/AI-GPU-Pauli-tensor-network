#!/bin/bash
# run_seismic_16x32.sh — Seismic tomography for 16x32 (512-qubit) lattice.
#
# Centers: 8x8 Voronoi tile [5,10,23,28,33,46,51] tiled 2x4 across 16x32.
# 56 centers total, chi=40 (fits at ~53% VRAM, no v_hop needed for seismic).
#
# Protocol (same as 7x7 adaptive):
#   1. Seismic: all 56 centers fire simultaneously (2 runs: base + pert)
#   2. Detection: detect_delta_logT.py on seismic logs
#   3. If wrong: aperture with nearest center (v_hop=0.5, chi=50)
#
# Usage:
#   bash scripts/run_seismic_16x32.sh <bond> <J_pert> [trial_id]
#   bash scripts/run_seismic_16x32.sh 200 0.800 1

BOND=${1:-200}
J_PERT=${2:-0.800}
TRIAL=${3:-1}

cd /scratch/MPF/CppSim

GRAPH="data/graphs/rect_16x32.json"
BINARY="./main_pauli_sweep_rect16x32"
CHI=40
STEPS=3
CUTOFF=1e-3
LOGDIR="logs/seismic_16x32/trial_${TRIAL}"
mkdir -p $LOGDIR

# 8x8 Voronoi tile [5,10,23,28,33,46,51] tiled 2 rows x 4 cols across 16x32
CENTERS="5,13,21,29,34,42,50,58,71,79,87,95,100,108,116,124,129,137,145,153,166,174,182,190,195,203,211,219,261,269,277,285,290,298,306,314,327,335,343,351,356,364,372,380,385,393,401,409,422,430,438,446,451,459,467,475"

PERTURB="${BOND},${J_PERT}"

echo "================================================================"
echo "=== Seismic 16x32  Trial=$TRIAL  Bond=$BOND  J_pert=$J_PERT ==="
echo "================================================================"
echo "Chi=$CHI  Steps=$STEPS  Centers=56  2-GPU parallel"
echo ""

# Base and pert run in parallel on GPU 0 and GPU 1
echo "--- Seismic base (GPU 0) + pert (GPU 1) in parallel ---"
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
grep "^STEP" $LOGDIR/seismic_base.log | head -5
grep "^STEP" $LOGDIR/seismic_pert.log | head -5

# Detection
echo ""
echo "--- Detection ---"
python3 scripts/detect_delta_logT.py \
    --logdir $LOGDIR \
    --graph $GRAPH \
    --broken logs/mawgm_16x32/trial_${TRIAL}/broken.json \
    --top_k 1 --thr 0.01

echo ""
echo "Done. Logs: $LOGDIR"
