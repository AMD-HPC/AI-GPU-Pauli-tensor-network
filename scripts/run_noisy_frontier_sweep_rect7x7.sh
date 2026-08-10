#!/bin/bash
# run_noisy_frontier_sweep_rect7x7.sh — Frontier sweep for rect_7x7 topology.
# Same structure as run_noisy_frontier_sweep.sh but uses main_pauli_sweep_rect7x7.
# Synthetic depolarizing noise (p=0.01*gamma).
#
# Usage: bash scripts/run_noisy_frontier_sweep_rect7x7.sh [n_steps] [center]

N=${1:-10}
CENTER=${2:-24}  # center of 7x7 grid (site 24 = position (3,3))
THRESHOLD=1e-4
echo "rect_7x7 frontier sweep: n_steps=$N center=$CENTER threshold=$THRESHOLD"

GRAPH="data/graphs/rect_7x7.json"

COMBOS=(
    "20   1e-3"
    "20   5e-4"
    "20   1e-4"
    "50   1e-3"
    "50   5e-4"
    "50   1e-4"
    "70   1e-3"
    "70   5e-4"
    "70   1e-5"
)

# Launch 3 at a time on 3 GPUs
pids=()
gpus=(0 1 2 0 1 2 0 1 2)
idx=0
for combo in "${COMBOS[@]}"; do
    chi=$(echo $combo | awk '{print $1}')
    delta=$(echo $combo | awk '{print $2}')
    log="logs/spatial_mpf_rect7x7_chi${chi}_${delta}.log"
    gpu=${gpus[$idx]}
    echo "Launching rect_7x7 chi=$chi cutoff=$delta on GPU $gpu → $log"
    ROCR_VISIBLE_DEVICES=$gpu \
        ./main_pauli_sweep_rect7x7 $chi $N $delta --center=$CENTER \
        2>/dev/null > "$log" &
    pids+=($!)
    idx=$((idx+1))
    if [ ${#pids[@]} -eq 3 ]; then
        echo "Waiting for batch..."
        for pid in "${pids[@]}"; do wait $pid; done
        pids=()
    fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "All rect_7x7 runs done."

echo ""
echo "=== Frontier analysis (threshold=$THRESHOLD) ==="
printf "%-35s  %5s  %5s  %5s  %5s\n" "Config" "F_nl" "F_n1" "F_n2" "F_n3"
printf "%s\n" "$(printf '─%.0s' {1..60})"

for combo in "${COMBOS[@]}"; do
    chi=$(echo $combo | awk '{print $1}')
    delta=$(echo $combo | awk '{print $2}')
    log="logs/spatial_mpf_rect7x7_chi${chi}_${delta}.log"
    tag="rect7x7_chi${chi}_${delta}"

    for label in NL N1 N2 N3; do
        grep "PAULI_${label}" "$log" | sed "s/PAULI_${label}/PAULI/" \
            > /tmp/${tag}_${label}.dat 2>/dev/null
    done

    get_f() {
        python3 scripts/frontier_at_cycle.py \
            --dat "$1" --graph "$GRAPH" \
            --center $CENTER --threshold $THRESHOLD 2>/dev/null | \
            tail -1 | awk '{print $2}'
    }

    F_nl=$(get_f /tmp/${tag}_NL.dat)
    F_n1=$(get_f /tmp/${tag}_N1.dat)
    F_n2=$(get_f /tmp/${tag}_N2.dat)
    F_n3=$(get_f /tmp/${tag}_N3.dat)

    printf "%-35s  %5s  %5s  %5s  %5s\n" "$tag" "$F_nl" "$F_n1" "$F_n2" "$F_n3"
done
