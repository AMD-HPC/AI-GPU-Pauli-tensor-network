#!/bin/bash
# run_10_mawgm.sh — Run 10 MAWGM trials
#
# Usage:
#   bash scripts/run_10_mawgm.sh [-0.20] [1]
#   bash scripts/run_10_mawgm.sh -0.20 2   # 2 broken bonds

DJ=${1:--0.20}
N_BONDS=${2:-1}
SEED=42

cd /scratch/MPF/CppSim
mkdir -p logs/mawgm_v2

echo "=== 10 MAWGM Trials (3 scans, 3 GPUs parallel, d_max=2, 5 cycles) ===" | tee logs/mawgm_v2/summary.txt
echo "dJ=$DJ  n_bonds=$N_BONDS  seed=$SEED  $(date)" | tee -a logs/mawgm_v2/summary.txt
echo "Scans: corners+center / quadrants / mid-edges — centers 1-based, binary -1 inside" | tee -a logs/mawgm_v2/summary.txt
echo "" | tee -a logs/mawgm_v2/summary.txt
printf "%-8s %-8s %-12s %-10s %-8s\n" \
    "Trial" "Bond" "Sites" "Contrast" "Result" | tee -a logs/mawgm_v2/summary.txt
echo "----------------------------------------------------" | tee -a logs/mawgm_v2/summary.txt

CORRECT=0
for TRIAL in $(seq 1 10); do
    bash scripts/run_mawgm.sh $TRIAL $SEED $DJ $N_BONDS > /tmp/mawgm_${TRIAL}.log 2>&1

    RESULT=$(python3 -c "
import json
try:
    r = json.load(open('logs/mawgm_v2/trial_${TRIAL}/result.json'))
    tag = '✓ CORRECT' if r['correct'] else '✗ WRONG'
    print(f\"{r['declared_bond']:<8} {r['declared_sites'][0]}↔{r['declared_sites'][1]:<8} {r['contrast']:.4f}     {tag}\")
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null)

    printf "%-8s %s\n" "Trial $TRIAL" "$RESULT" | tee -a logs/mawgm_v2/summary.txt
    echo "$RESULT" | grep -q "CORRECT" && CORRECT=$((CORRECT+1))
done

echo "" | tee -a logs/mawgm_v2/summary.txt
echo "=== Final: $CORRECT/10 correct ===" | tee -a logs/mawgm_v2/summary.txt
echo "=== DONE $(date) ===" | tee -a logs/mawgm_v2/summary.txt
