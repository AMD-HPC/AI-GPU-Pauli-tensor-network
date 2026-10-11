#!/bin/bash
# run_10trials.sh — Run 10 adaptive tomography trials sequentially
#
# Usage:
#   bash scripts/run_10trials.sh [-0.20]
#
# Results summary in logs/adaptive_tomo/summary.txt

DJ=${1:--0.20}
N_BONDS=${2:-1}
SEED=42

cd /scratch/MPF/CppSim
LOGBASE="logs/adaptive_tomo_${N_BONDS}b"
mkdir -p $LOGBASE

echo "=== 10 Adaptive Tomography Trials (Voronoi k=7, n_bonds=$N_BONDS) ===" | tee $LOGBASE/summary.txt
echo "dJ=$DJ  seed_base=$SEED  $(date)" | tee -a $LOGBASE/summary.txt
echo "Centers: 3,13,15,25,35,37,47 (ties=0, max_T_b=5, 5 cycles)" | tee -a $LOGBASE/summary.txt
echo "" | tee -a logs/adaptive_tomo2/summary.txt
printf "%-8s %-10s %-12s %-10s %-12s %-10s\n" \
    "Trial" "Bonds" "Endpoints" "Seismic" "Aperture" "Method" | tee -a $LOGBASE/summary.txt
echo "--------------------------------------------------------------------" | tee -a $LOGBASE/summary.txt

PASS1=0; PASS2=0; PASS_BOTH=0

for TRIAL in $(seq 1 10); do
    bash scripts/run_adaptive_tomo.sh $TRIAL $SEED $DJ "" $N_BONDS > /tmp/trial_${TRIAL}.log 2>&1

    # Extract result
    RESULT=$(python3 -c "
import json
try:
    r = json.load(open('${LOGBASE}/trial_${TRIAL}/result.json'))
    s = '✓' if r['seismic_correct'] else '✗'
    a = '✓' if r.get('aperture_correct') else ('skip' if not r['aperture_run'] else '✗')
    m = r.get('method','?')
    bonds = r.get('bond', '?')
    print(f\"{str(bonds):<10} {r['true_a']}↔{r['true_b']:<8} {s:<10} {a:<12} {m}\")
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null)

    printf "%-8s %s\n" "Trial $TRIAL" "$RESULT" | tee -a $LOGBASE/summary.txt
done

echo "" | tee -a $LOGBASE/summary.txt
echo "=== Summary ===" | tee -a $LOGBASE/summary.txt
python3 -c "
import json, glob
results = [json.load(open(f)) for f in sorted(glob.glob('${LOGBASE}/trial_*/result.json'))]
n = len(results)
s_correct = sum(r['seismic_correct'] for r in results)
a_run = sum(r['aperture_run'] for r in results)
a_correct = sum(r.get('aperture_correct',False) for r in results)
final = sum(r['final_correct'] for r in results)
print(f'  Seismic correct (no aperture needed): {s_correct}/{n}')
print(f'  Aperture run: {a_run}/{n}')
print(f'  Aperture correct: {a_correct}/{a_run if a_run else 1}')
print(f'  Final correct (seismic+aperture): {final}/{n}')
" 2>/dev/null | tee -a $LOGBASE/summary.txt

echo "=== DONE $(date) ===" | tee -a $LOGBASE/summary.txt
