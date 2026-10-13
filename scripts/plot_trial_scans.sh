#!/bin/bash
# plot_trial_scans.sh — Plot all scans for a given trial
#
# Usage:
#   bash scripts/plot_trial_scans.sh <logbase> <trial> [max_cycles] [lmin]
#   bash scripts/plot_trial_scans.sh logs/mawgm_f64_2b 4

LOGBASE=${1:-logs/mawgm_f64_2b}
TRIAL=${2:-4}
MAX_CYCLES=${3:-3}
LMIN=${4:-0.02}

GRAPH="data/graphs/rect_7x7.json"
PLAN="data/scans/rect_7x7_scan_3gpu.json"
LOGDIR="${LOGBASE}/trial_${TRIAL}"

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate 2>/dev/null || true

echo "Plotting trial ${TRIAL} from ${LOGBASE}..."

for S in 1 2 3; do
    CENTERS=$(python3 -c "
import json
p = json.load(open('${PLAN}'))
print(','.join(map(str, p['placements'][${S}-1]['centers'])))
")
    OUT="${LOGDIR}/scan${S}_cycles.png"
    python3 scripts/plot_seismic_cycles.py \
        --baseline  ${LOGDIR}/scan${S}_base.log \
        --perturbed ${LOGDIR}/scan${S}_pert.log \
        --graph ${GRAPH} \
        --broken ${LOGDIR}/broken.json \
        --centers ${CENTERS} \
        --max-cycles ${MAX_CYCLES} \
        --thr 1e-3 \
        --lmin ${LMIN} \
        --out ${OUT}
    echo "  Saved: ${OUT}"
done

echo "Done."
