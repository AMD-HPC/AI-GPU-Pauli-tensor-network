#!/bin/bash
# run_targeted_aperture.sh — Run targeted aperture validations for trial 5
# 2 cycles + v_hop=0.448 — tiny chi, very fast
# All 5 candidate bonds, 3 GPUs

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate 2>/dev/null || true

LOGDIR="logs/mawgm_f64_2b/trial_5"
BINARY="./main_pauli_sweep_rect7x7"
CHI=50; STEPS=5; CUTOFF=1e-3
PERTURB="--perturb-bond=8;45,0.800"
V_HOP=""   # no v_hop — targeted aperture uses full grid

run_apt() {
    local BOND=$1; local CENTER=$2; local GPU=$3
    local BASE="$LOGDIR/apt_bond${BOND}_c${CENTER}_base.log"
    local PERT="$LOGDIR/apt_bond${BOND}_c${CENTER}_pert.log"
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTER --state=NL $V_HOP > $BASE 2>&1
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$CENTER --state=NL $V_HOP "$PERTURB" > $PERT 2>&1
    echo "  Bond $BOND via center $CENTER done."
}

echo "=== Targeted aperture validation — trial 5 ==="
echo "2 cycles, v_hop=0.448, 3 GPUs parallel"
echo "$(date)"

# 5 candidates: bonds 45,2,3,46,8 — centers 2,1,2,3,2
# GPU0: bonds 45 and 2
# GPU1: bonds 3 and 46
# GPU2: bond 8
( run_apt 45 2 0; run_apt 2 1 0 ) &
( run_apt 3  2 1; run_apt 46 3 1 ) &
( run_apt 8  2 2 ) &
wait

echo "All done. $(date)"
echo "Now validate with:"
echo "python3 scripts/targeted_aperture.py --logdir $LOGDIR --broken $LOGDIR/broken.json --mode validate --top_k 5"
