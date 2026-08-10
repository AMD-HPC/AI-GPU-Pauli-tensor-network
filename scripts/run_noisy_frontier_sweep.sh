#!/bin/bash
# run_noisy_frontier_sweep.sh — Sweep chi and delta for noisy+noiseless frontier comparison.
#
# Goal: find (chi, delta) where noisy frontier reaches same distance as noiseless.
# If found → simulation is faithful, amplitude difference is real physics.
# If not found → noise kills amplitude below threshold regardless of chi/delta.
#
# Uses main_heavyhex_gamma_mpf_2gpu --measure-all (outputs PAULI_NL + PAULI_N1/N2/N3).
# Analyzes with analyze_lightcone.py on each gamma separately.
#
# Usage: bash scripts/run_noisy_frontier_sweep.sh [n_steps] [center]

N=${1:-10}
CENTER=${2:-13}
THRESHOLD=1e-4
LOGDIR="logs/noisy_frontier_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR  (n_steps=$N center=$CENTER threshold=$THRESHOLD)"

# Parameter combinations: chi delta
COMBOS=(
    "70   1e-3"
    "70   5e-4"
    "70   1e-4"
    "200  1e-3"
    "200  5e-4"
    "200  1e-4"
)

GRAPH="data/graphs/heavy_hex_3x3.json"

# Each run uses 1 GPU — run 3 in parallel, then next batch
pids=()
gpus=(0 1 2 0 1 2)
idx=0
for combo in "${COMBOS[@]}"; do
    chi=$(echo $combo | awk '{print $1}')
    delta=$(echo $combo | awk '{print $2}')
    tag="chi${chi}_cut${delta}"
    log="logs/spatial_mpf_chi${chi}_${delta}.log"
    gpu=${gpus[$idx]}
    echo "Launching chi=$chi cutoff=$delta on GPU $gpu → $log"
    ROCR_VISIBLE_DEVICES=$gpu \
        ./main_heavyhex_gamma_mpf_2gpu $chi $N $delta --center=$CENTER --measure-all \
        2>/dev/null > "$log" &
    pids+=($!)
    idx=$((idx+1))
    # 3 GPUs → run 3 at a time
    if [ ${#pids[@]} -eq 3 ]; then
        echo "Waiting for batch..."
        for pid in "${pids[@]}"; do wait $pid; done
        pids=()
    fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "All runs done."

echo ""
echo "=== Frontier analysis (threshold=$THRESHOLD) ==="
echo ""
printf "%-20s  %6s  %6s  %6s  %6s  %6s\n" "Config" "F_nl" "F_n1" "F_n2" "F_n3" "Same?"

for combo in "${COMBOS[@]}"; do
    chi=$(echo $combo | awk '{print $1}')
    delta=$(echo $combo | awk '{print $2}')
    tag="chi${chi}_cut${delta}"
    log="$LOGDIR/${tag}.log"

    # Extract frontier per gamma using analyze_lightcone.py
    # Create per-gamma PAULI files
    for label in NL N1 N2 N3; do
        grep "PAULI_${label}" "$log" | sed "s/PAULI_${label}/PAULI/" \
            > "$LOGDIR/${tag}_${label}.dat"
        # Add STEP lines for C_nl (needed by analyze_lightcone)
        grep "^STEP" "$log" >> "$LOGDIR/${tag}_${label}.dat"
    done

    # Get frontier at last cycle for each gamma
    get_frontier_last() {
        local datfile=$1
        python3 scripts/frontier_at_cycle.py \
            --dat "$datfile" \
            --graph "$GRAPH" \
            --center $CENTER \
            --threshold $THRESHOLD 2>/dev/null | \
            tail -1 | awk '{print $2}'
    }

    F_nl=$(get_frontier_last "$LOGDIR/${tag}_NL.dat")
    F_n1=$(get_frontier_last "$LOGDIR/${tag}_N1.dat")
    F_n2=$(get_frontier_last "$LOGDIR/${tag}_N2.dat")
    F_n3=$(get_frontier_last "$LOGDIR/${tag}_N3.dat")

    # Check if all frontiers match noiseless
    same="no"
    if [ "$F_nl" = "$F_n1" ] && [ "$F_nl" = "$F_n2" ] && [ "$F_nl" = "$F_n3" ]; then
        same="YES"
    fi

    printf "%-20s  %6s  %6s  %6s  %6s  %6s\n" \
        "$tag" "$F_nl" "$F_n1" "$F_n2" "$F_n3" "$same"
done

echo ""
echo "All done. Logs in $LOGDIR"
echo "To plot: python3 scripts/analyze_spatial_mpf.py --log $LOGDIR/chi70_d1e-3.log ..."
