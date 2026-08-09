#!/bin/bash
# run_dg_sweep.sh — Sweep Trotter cutoff and eig_cut for DG gauge comparison
#
# Usage: bash run_dg_sweep.sh [chi_max] [n_steps]

CHI=${1:-250}
N=${2:-10}
CENTER=13
LOGDIR="logs/dg_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR"

# Trotter cutoffs to sweep (delta)
DELTAS="1e-3 1e-4 1e-5"

# Eig cutoffs to sweep (fixed delta=1e-3 for this sub-sweep)
EIG_CUTS="1e-1 1e-2 1e-3"

echo "=== Phase 1: vary Trotter delta (eig_cut=1e-2 fixed) ==="
for delta in $DELTAS; do
    tag="delta${delta}_ec1e-2"
    log="$LOGDIR/DG_${tag}.log"
    echo "Running DG chi=$CHI n=$N delta=$delta eig_cut=1e-2 ..."
    ./main_heavyhex_gamma_mpf_DG_2gpu $CHI $N $delta \
        --center=$CENTER --eig-cut=1e-2 2>/dev/null > "$log"
    echo "  Done: $log"
done

echo ""
echo "=== Phase 2: vary eig_cut (delta=1e-3 fixed) ==="
for ec in $EIG_CUTS; do
    tag="delta1e-3_ec${ec}"
    log="$LOGDIR/DG_${tag}.log"
    echo "Running DG chi=$CHI n=$N delta=1e-3 eig_cut=$ec ..."
    ./main_heavyhex_gamma_mpf_DG_2gpu $CHI $N 1e-3 \
        --center=$CENTER --eig-cut=$ec 2>/dev/null > "$log"
    echo "  Done: $log"
done

echo ""
echo "=== Reference: QR at each delta ==="
for delta in $DELTAS; do
    tag="delta${delta}"
    log="$LOGDIR/QR_${tag}.log"
    echo "Running QR chi=$CHI n=$N delta=$delta ..."
    ./main_heavyhex_gamma_mpf_2gpu $CHI $N $delta \
        --center=$CENTER 2>/dev/null > "$log"
    echo "  Done: $log"
done

echo ""
echo "All done. Logs in $LOGDIR"
echo ""
echo "To analyze:"
echo "  python3 analyze_dg_sweep.py --dir $LOGDIR"
