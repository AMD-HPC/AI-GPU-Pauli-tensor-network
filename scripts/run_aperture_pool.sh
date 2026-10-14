#!/bin/bash
# run_aperture_pool.sh — GPU pool aperture validation
#
# Each aperture (base+pert) runs on the SAME GPU for HW consistency.
# Different bonds run in parallel on different GPUs.
# Center ON bond endpoint + v_hop=0.448 + 3 cycles = lightning fast.
#
# Usage:
#   bash scripts/run_aperture_pool.sh <logdir> <bond_tag> <j_new> <candidates>
#   # candidates = "bond:center bond:center ..."
#   # Example:
#   bash scripts/run_aperture_pool.sh logs/mawgm_f64_2b/trial_1 "4;36" 0.800 "4:5 36:29 77:22 46:5 59:4"

LOGDIR=$1
BOND_TAG=$2
J_NEW=$3
CANDIDATES=$4   # space-separated "bond:center" pairs

BINARY="./main_pauli_sweep_rect7x7"
CHI=50; STEPS=3; CUTOFF=1e-3
V_HOP="--v_hop=0.448"

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate 2>/dev/null || true

echo "=== Aperture pool: $LOGDIR ==="
echo "Bonds: $BOND_TAG  J_new: $J_NEW"
echo "$(date)"

# Run one aperture: base+pert on same GPU
run_aperture() {
    local BOND=$1; local CENTER=$2; local GPU=$3
    local BASE="$LOGDIR/apt_b${BOND}_c${CENTER}_base.log"
    local PERT="$LOGDIR/apt_b${BOND}_c${CENTER}_pert.log"
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTER --state=NL $V_HOP > $BASE 2>&1
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTER --state=NL $V_HOP \
        "--perturb-bond=${BOND_TAG},${J_NEW}" > $PERT 2>&1
    echo "  Bond $BOND center $CENTER GPU$GPU done"
}

# Assign GPUs round-robin, 3 at a time
GPU=0
PIDS=()
for BC in $CANDIDATES; do
    BOND=$(echo $BC | cut -d: -f1)
    CENTER=$(echo $BC | cut -d: -f2)
    run_aperture $BOND $CENTER $GPU &
    PIDS+=($!)
    GPU=$(( (GPU+1) % 3 ))
    # When all 3 GPUs busy, wait for them to finish
    if [ $GPU -eq 0 ]; then
        wait "${PIDS[@]}"
        PIDS=()
    fi
done
# Wait for remaining
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"

echo "All apertures done. $(date)"
