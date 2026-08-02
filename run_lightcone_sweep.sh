#!/bin/bash
# run_lightcone_sweep.sh — Parallel lightcone sweep across chi and cutoff
#
# Runs main_lightcone on multiple GPUs simultaneously.
# Each combination gets its own GPU via ROCR_VISIBLE_DEVICES.
#
# Usage: bash run_lightcone_sweep.sh [n_steps]

N_STEPS=${1:-20}
LOGDIR="logs/lightcone_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR"

# Combinations: chi cutoff gpu
RUNS=(
    "70  2e-3  0"
    "70  1e-3  1"
    "70  5e-4  2"
    "200 2e-3  0"
    "200 1e-3  1"
    "200 5e-4  2"
)

pids=()
for run in "${RUNS[@]}"; do
    chi=$(echo $run | awk '{print $1}')
    cut=$(echo $run | awk '{print $2}')
    gpu=$(echo $run | awk '{print $3}')
    tag="chi${chi}_cut${cut}"
    logfile="$LOGDIR/${tag}.dat"
    echo "Launching chi=$chi cutoff=$cut → GPU $gpu → $logfile"
    ROCR_VISIBLE_DEVICES=$gpu ./main_lightcone $chi $N_STEPS $cut 2>/dev/null > "$logfile" &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} runs..."
for pid in "${pids[@]}"; do
    wait $pid
done
echo "All done. Logs in $LOGDIR"

# Analyze each
GRAPH="/scratch/MPF/CppSim/data/graphs/heavy_hex_3x3.json"
for run in "${RUNS[@]}"; do
    chi=$(echo $run | awk '{print $1}')
    cut=$(echo $run | awk '{print $2}')
    tag="chi${chi}_cut${cut}"
    logfile="$LOGDIR/${tag}.dat"
    outpng="$LOGDIR/${tag}_analysis.png"
    echo "=== chi=$chi cut=$cut ==="
    python3 analyze_lightcone.py --dat "$logfile" --graph "$GRAPH" \
        --center 2 --threshold 1e-4 --out "$outpng" 2>/dev/null | \
        grep -E "Frontier velocity|plateau|Graph:"
done
