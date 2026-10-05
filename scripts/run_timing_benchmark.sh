#!/bin/bash
# run_timing_benchmark.sh — Timing: Aperture vs Seismic on 7x7
#
# Seismic:  GPU0=baseline, GPU1=perturbed — 2 GPUs in parallel
# Aperture: 6 centers, 3 GPUs as pool (2 centers per GPU sequential)
#
# Usage:
#   bash scripts/run_timing_benchmark.sh [bond] [delta_J] [chi] [steps]
#   bash scripts/run_timing_benchmark.sh 12 0.20 50 5

BOND=${1:-12}
DJ=${2:-0.20}
CHI=${3:-50}
STEPS=${4:-5}
J_NEW=$(python3 -c "print(f'{1.0 + $DJ:.3f}')")

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

LOGDIR="logs/timing_benchmark"
mkdir -p $LOGDIR

GRAPH="data/graphs/rect_7x7.json"
BINARY="./main_pauli_sweep_rect7x7"
CUTOFF=1e-3
CENTERS_STR="25,10,34,30,6,32"
CENTERS=(25 10 34 30 6 32)

echo "=== Timing Benchmark: Aperture vs Seismic ===" | tee $LOGDIR/timing.txt
echo "Topology: rect 7x7  Bond=$BOND  dJ=$DJ  J_new=$J_NEW" | tee -a $LOGDIR/timing.txt
echo "Chi=$CHI  Steps=$STEPS  Cutoff=$CUTOFF" | tee -a $LOGDIR/timing.txt
echo "$(date)" | tee -a $LOGDIR/timing.txt
echo "" | tee -a $LOGDIR/timing.txt

# C_breaker config
python3 scripts/breaker.py \
    --graph $GRAPH --bond $BOND --coupling $DJ \
    --out $LOGDIR/broken.json > /dev/null 2>&1

# ── SEISMIC: GPU0=baseline, GPU1=perturbed, parallel ──────────────────────────
echo "=== SEISMIC ===" | tee -a $LOGDIR/timing.txt
echo "  GPU0: baseline  GPU1: perturbed  (parallel)"
T_SEISMIC_START=$(date +%s%N)

ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
    --center=$CENTERS_STR --state=NL \
    > $LOGDIR/seismic_base.log 2>&1 &
PID_S0=$!

ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
    --center=$CENTERS_STR --state=NL \
    --perturb-bond=${BOND},${J_NEW} \
    > $LOGDIR/seismic_pert.log 2>&1 &
PID_S1=$!

wait $PID_S0 $PID_S1
T_SEISMIC_END=$(date +%s%N)
T_SEISMIC=$(( (T_SEISMIC_END - T_SEISMIC_START) / 1000000 ))

# Parse seismic timing from logs
S0_TIME=$(grep "^STEP $STEPS" $LOGDIR/seismic_base.log | awk '{print $NF}' | tail -1)
S1_TIME=$(grep "^STEP $STEPS" $LOGDIR/seismic_pert.log | awk '{print $NF}' | tail -1)
echo "  Baseline wall: ${S0_TIME}s  Perturbed wall: ${S1_TIME}s" | tee -a $LOGDIR/timing.txt
echo "  Total wall (parallel): ${T_SEISMIC}ms" | tee -a $LOGDIR/timing.txt

# Search
python3 scripts/searcher.py \
    --baseline  $LOGDIR/seismic_base.log \
    --perturbed $LOGDIR/seismic_pert.log \
    --graph $GRAPH --centers $CENTERS_STR \
    --broken $LOGDIR/broken.json \
    --out $LOGDIR/seismic_result.json 2>&1 | grep -E "declares|Result|max L"

echo "" | tee -a $LOGDIR/timing.txt

# ── APERTURE: 3 GPUs, 2 centers per GPU sequential ────────────────────────────
echo "=== APERTURE ===" | tee -a $LOGDIR/timing.txt
echo "  GPU0: centers ${CENTERS[0]},${CENTERS[1]}  GPU1: centers ${CENTERS[2]},${CENTERS[3]}  GPU2: centers ${CENTERS[4]},${CENTERS[5]}"

T_APERTURE_START=$(date +%s%N)

run_aperture_pair() {
    local C1=$1 C2=$2 GPU=$3
    cd /scratch/MPF/CppSim

    # Center 1
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C1 --state=NL \
        > $LOGDIR/aperture_base_c${C1}.log 2>&1
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C1 --state=NL \
        --perturb-bond=${BOND},${J_NEW} \
        > $LOGDIR/aperture_pert_c${C1}.log 2>&1

    # Center 2
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C2 --state=NL \
        > $LOGDIR/aperture_base_c${C2}.log 2>&1
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C2 --state=NL \
        --perturb-bond=${BOND},${J_NEW} \
        > $LOGDIR/aperture_pert_c${C2}.log 2>&1
}

export BOND J_NEW CHI STEPS CUTOFF BINARY LOGDIR

( run_aperture_pair ${CENTERS[0]} ${CENTERS[1]} 0 ) &
( run_aperture_pair ${CENTERS[2]} ${CENTERS[3]} 1 ) &
( run_aperture_pair ${CENTERS[4]} ${CENTERS[5]} 2 ) &
wait

T_APERTURE_END=$(date +%s%N)
T_APERTURE=$(( (T_APERTURE_END - T_APERTURE_START) / 1000000 ))

echo "  Total wall (3 GPU pool, 2 centers per GPU sequential): ${T_APERTURE}ms" | tee -a $LOGDIR/timing.txt

# Per-center timing
echo "  Per-center simulation times:" | tee -a $LOGDIR/timing.txt
for C in ${CENTERS[@]}; do
    T=$(grep "^STEP $STEPS" $LOGDIR/aperture_base_c${C}.log | awk '{print $NF}' | tail -1)
    echo "    Center $C: ${T}s" | tee -a $LOGDIR/timing.txt
done

# Ensemble search
BASELINES=$(IFS=,; echo "${CENTERS[@]/#/logs/timing_benchmark/aperture_base_c}"); BASELINES="${BASELINES//$'\n'/}.log"
python3 scripts/searcher.py \
    --baseline  "$(IFS=,; printf 'logs/timing_benchmark/aperture_base_c%s.log,' "${CENTERS[@]}" | sed 's/,$//')" \
    --perturbed "$(IFS=,; printf 'logs/timing_benchmark/aperture_pert_c%s.log,' "${CENTERS[@]}" | sed 's/,$//')" \
    --graph $GRAPH --centers $CENTERS_STR \
    --broken $LOGDIR/broken.json \
    --out $LOGDIR/aperture_result.json 2>&1 | grep -E "declares|Result|max L"

echo "" | tee -a $LOGDIR/timing.txt

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=== SUMMARY ===" | tee -a $LOGDIR/timing.txt
echo "  Seismic  wall: ${T_SEISMIC}ms   (2 GPUs parallel)" | tee -a $LOGDIR/timing.txt
echo "  Aperture wall: ${T_APERTURE}ms  (3 GPUs, 2 centers/GPU sequential)" | tee -a $LOGDIR/timing.txt
SPEEDUP=$(python3 -c "print(f'{$T_APERTURE / max($T_SEISMIC,1):.2f}x')" 2>/dev/null || echo "?")
echo "  Aperture/Seismic ratio: ${SPEEDUP}" | tee -a $LOGDIR/timing.txt
echo "" | tee -a $LOGDIR/timing.txt
echo "=== DONE $(date) ===" | tee -a $LOGDIR/timing.txt
echo ""
echo "Results: $LOGDIR/"
echo "Timing:  $LOGDIR/timing.txt"
