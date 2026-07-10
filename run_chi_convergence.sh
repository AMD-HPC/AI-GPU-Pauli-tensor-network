#!/bin/bash
# run_chi_convergence.sh — Chi convergence study on heavy_hex_3x3
#
# Center: site 13 (mid-graph junction, symmetric spreading)
# 3 waves × 3 GPUs, chi = 50..430
#
# Usage: bash run_chi_convergence.sh [n_steps] [dt] [J] [cutoff]
#
# Wave 1: chi=50,  100, 150   (~fast, few minutes each)
# Wave 2: chi=200, 250, 300   (~medium, 10-30 min each)
# Wave 3: chi=350, 400, 430   (~slow, 30-90 min each)
#
# Output: logs/chi_convergence/spatial_chi{N}.dat  — Pauli weights per cycle
#         logs/chi_convergence/chi_{N}.log         — timing + C(t)

N_STEPS=${1:-30}
DT=${2:-0.1}
J=${3:-1.0}
CUTOFF=${4:-1e-4}
QR=${5:---qr=householder}
CENTER=13  # site 13: mid-graph, 3-fold symmetric spreading

mkdir -p logs/chi_convergence

echo "═══════════════════════════════════════════════════════════"
echo "  Chi convergence study — heavy_hex_3x3"
echo "  Center: site $CENTER  Steps: $N_STEPS  dt=$DT  J=$J  cutoff=$CUTOFF"
echo "  Chi values: 50 100 150 | 200 250 300 | 350 400 430"
echo "═══════════════════════════════════════════════════════════"

run_wave() {
    local chi1=$1 chi2=$2 chi3=$3
    echo ""
    echo "── Wave: chi=$chi1  chi=$chi2  chi=$chi3 ──"

    ROCR_VISIBLE_DEVICES=0 ./main_heavyhex $chi1 $N_STEPS $DT $J $CUTOFF $QR \
        --center=$CENTER \
        --spatial=logs/chi_convergence/spatial_chi${chi1}.dat \
        > logs/chi_convergence/chi_${chi1}.log 2>&1 &
    PID0=$!

    ROCR_VISIBLE_DEVICES=1 ./main_heavyhex $chi2 $N_STEPS $DT $J $CUTOFF $QR \
        --center=$CENTER \
        --spatial=logs/chi_convergence/spatial_chi${chi2}.dat \
        > logs/chi_convergence/chi_${chi2}.log 2>&1 &
    PID1=$!

    ROCR_VISIBLE_DEVICES=2 ./main_heavyhex $chi3 $N_STEPS $DT $J $CUTOFF $QR \
        --center=$CENTER \
        --spatial=logs/chi_convergence/spatial_chi${chi3}.dat \
        > logs/chi_convergence/chi_${chi3}.log 2>&1 &
    PID2=$!

    echo "PIDs: chi=$chi1:$PID0  chi=$chi2:$PID1  chi=$chi3:$PID2"
    wait $PID0; echo "  chi=$chi1 done (exit $?)"
    wait $PID1; echo "  chi=$chi2 done (exit $?)"
    wait $PID2; echo "  chi=$chi3 done (exit $?)"
}

# Wave 1 — fast
run_wave 50 100 150

# Wave 2 — medium
run_wave 200 250 300

# Wave 3 — slow
run_wave 350 400 430

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All done. Results in logs/chi_convergence/"
echo ""
echo "  Next: python3 plot_ct_convergence.py"
echo "        python3 plot_lightcone.py --data logs/chi_convergence/spatial_chi*.dat"
echo "═══════════════════════════════════════════════════════════"
