#!/bin/bash
# run_ip_sweep.sh — Sweep chi and cutoff for inner product MPF comparison
# chi:    50 100 150 200 250
# cutoff: 2e-2 1e-2 2e-3 1e-3 1e-4
# n_steps: 10

LOGDIR="logs/ip_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR"

CHI_LIST="50 100 150 200 250"
CUT_LIST="2e-2 1e-2 2e-3 1e-3 1e-4"
N_STEPS=10

for chi in $CHI_LIST; do
    for cut in $CUT_LIST; do
        tag="chi${chi}_cut${cut}"
        logfile="$LOGDIR/${tag}.log"
        echo "Running chi=$chi cutoff=$cut → $logfile"
        ./main_heavyhex_gamma_mpf_2gpu $chi $N_STEPS $cut 2>/dev/null > "$logfile"
        echo "  done: $(tail -1 $logfile | grep -o 'STEP [0-9]*' | tail -1)"
    done
done

echo "All done. Logs in $LOGDIR"
echo "Run: python3 plot_ip_sweep.py --logdir $LOGDIR"
