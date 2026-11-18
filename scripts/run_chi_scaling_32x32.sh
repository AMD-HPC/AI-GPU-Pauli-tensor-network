#!/bin/bash
# run_chi_scaling_32x32.sh — Chi scaling study on 32x32 (1024 qubits)
#
# Three strategies compared:
#   A) Single-GPU seismic, no v_hop — OOMs at chi~40
#   B) Single-GPU seismic, v_hop=0.9 — stays tiny, can push chi high
#   C) 2-GPU seismic, no v_hop — fits to chi~45
#
# For each chi: run scan 1 base+pert, detect bond, record score/rank/VRAM.
#
# Usage:
#   bash scripts/run_chi_scaling_32x32.sh

cd /scratch/MPF/CppSim

BINARY1="./main_pauli_sweep_rect32x32"
BINARY2="./main_pauli_sweep_2gpu_rect32x32"
GRAPH="data/graphs/rect_32x32.json"
STEPS=3; CUTOFF=1e-3
BOND=200; J_PERT=0.800  # broken bond in GPU0 domain (row 6)
PERTURB="${BOND},${J_PERT}"

# Period-6 scan 1 centers for 32x32
# Rows 0,6,12,18,24,30 × cols 0,6,12,18,24,30
SCAN1="1,7,13,19,25,31,193,199,205,211,217,223,385,391,397,403,409,415,577,583,589,595,601,607,769,775,781,787,793,799,961,967,973,979,985,991"

LOGBASE="logs/chi_scaling_32x32"
mkdir -p $LOGBASE
SUMMARY="$LOGBASE/summary.tsv"
echo -e "chi\tstrategy\tVRAM%\tstep3_t\tscore\trank\tresult" > $SUMMARY

echo "================================================================"
echo "=== Chi Scaling Study: 32x32 (1024 qubits)  Bond=$BOND ==="
echo "================================================================"
echo "Strategy A: 1-GPU no v_hop   (OOM at chi~40)"
echo "Strategy B: 1-GPU v_hop=0.9  (memory tiny, chi-independent detection)"
echo "Strategy C: 2-GPU no v_hop   (fits to chi~45)"
echo ""
printf "%-6s %-10s %-8s %-8s %-8s %-8s %-8s\n" "chi" "strategy" "VRAM%" "step3_t" "score" "rank" "result"
echo "----------------------------------------------------------------------"

run_and_detect() {
    local CHI=$1; local STRATEGY=$2; local FLAGS=$3; local BINARY=$4
    local DIR="$LOGBASE/chi${CHI}_${STRATEGY}"
    mkdir -p $DIR

    # Run base
    ROCR_VISIBLE_DEVICES=0,1,2 $BINARY $CHI $STEPS $CUTOFF \
        --center=$SCAN1 $FLAGS \
        > $DIR/scan1_base.log 2>&1
    # Check OOM
    if grep -q "OOM\|out of memory" $DIR/scan1_base.log; then
        printf "%-6s %-10s %-8s %-8s %-8s %-8s %-8s\n" "$CHI" "$STRATEGY" "OOM" "-" "-" "-" "OOM"
        echo -e "$CHI\t$STRATEGY\tOOM\t-\t-\t-\tOOM" >> $SUMMARY
        return
    fi

    # Run pert
    ROCR_VISIBLE_DEVICES=0,1,2 $BINARY $CHI $STEPS $CUTOFF \
        --center=$SCAN1 $FLAGS \
        --perturb-bond=$PERTURB \
        > $DIR/scan1_pert.log 2>&1

    # Get timing
    T=$(grep "^STEP 3" $DIR/scan1_base.log | awk '{print $NF}')

    # Detect
    RESULT=$(python3 scripts/detect_delta_logT.py \
        --logdir $DIR --graph $GRAPH \
        --top_k 1 --thr 0.01 2>/dev/null | grep "^Result:\|bond $BOND rank")
    RANK=$(echo "$RESULT" | grep "bond $BOND rank" | grep -oP "rank: \K[0-9]+")
    SCORE=$(python3 scripts/detect_delta_logT.py \
        --logdir $DIR --graph $GRAPH \
        --top_k 1 --thr 0.01 2>/dev/null | grep "^\s*1 " | awk '{print $4}')
    FOUND=$(echo "$RESULT" | grep -c "FOUND\|BOTH")
    TAG=$([ "$FOUND" -gt "0" ] 2>/dev/null && echo "✓" || echo "✗")

    printf "%-6s %-10s %-8s %-8s %-8s %-8s %-8s\n" "$CHI" "$STRATEGY" "?" "$T" "${SCORE:-0}" "${RANK:-?}" "$TAG"
    echo -e "$CHI\t$STRATEGY\t?\t$T\t${SCORE:-0}\t${RANK:-?}\t$TAG" >> $SUMMARY
}

# Chi sweep
for CHI in 10 20 30 40 50; do
    run_and_detect $CHI "A_1gpu"    ""             "$BINARY1"
    run_and_detect $CHI "B_vhop09" "--v_hop=0.9"  "$BINARY1"
    run_and_detect $CHI "C_2gpu"   ""             "$BINARY2"
    echo ""
done

echo "Done. Logs: $LOGBASE"
