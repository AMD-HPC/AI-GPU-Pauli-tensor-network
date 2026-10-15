#!/bin/bash
# run_full_pipeline.sh — Full MAWGM + targeted aperture pipeline on 10 trials
#
# Phase 1: 3-scan MAWGM (already done in logs/mawgm_f64_2b/)
# Phase 2: Transfer ratio candidate generation
# Phase 3: Targeted aperture for top-k candidates (2 simulations per candidate)
# Phase 4: Validation
#
# Usage: bash scripts/run_full_pipeline.sh

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate 2>/dev/null || true

LOGBASE="logs/mawgm_f64_2b"
BINARY="./main_pauli_sweep_rect7x7"
CHI=50; STEPS=7; CUTOFF=1e-3  # 7 cycles: enough even with broken bonds attenuating
export CHI STEPS CUTOFF BINARY
TOP_K=5
SEED=42

echo "=== Full pipeline: MAWGM + targeted aperture ==="
echo "$(date)"
echo ""

correct=0; total=0

for TRIAL in $(seq 1 10); do
    LOGDIR="${LOGBASE}/trial_${TRIAL}"
    BROKEN="${LOGDIR}/broken.json"
    if [ ! -f "$BROKEN" ]; then echo "Trial $TRIAL: no broken.json, skip"; continue; fi

    # Get perturbation info
    BOND_TAG=$(python3 -c "
import json; br=json.load(open('$BROKEN'))
bonds=br.get('bonds',[br.get('bond',-1)])
print(';'.join(map(str,sorted(bonds))))
")
    J_NEW=$(python3 -c "
import json; br=json.load(open('$BROKEN'))
print(f\"{br.get('perturbation',{}).get('J_perturbed',0.8):.3f}\")
")

    echo "--- Trial $TRIAL: bonds=$BOND_TAG J_new=$J_NEW ---"

    # Phase 2+3: Python outputs "bond_idx center" pairs, bash runs simulations
    ASSIGNMENTS=$(python3 -c "
import sys, json, os
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites
from bfs_transfer_detect import bfs_transfer_detect
from targeted_aperture import find_optimal_center

g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}
plan = json.load(open('data/scans/rect_7x7_scan_3gpu.json'))

ranked, _ = bfs_transfer_detect('$LOGDIR', plan, g, adj, all_sites, site_idx,
                                  thr=1e-3, ratio_thr=1.05, T_max=7, verbose=False)
candidates = [(bi, score) for bi, score in ranked[:$TOP_K]]

for bi, score in candidates:
    a, b = get_bond_sites(g['edges'][bi])
    center, endpoint, cscore = find_optimal_center(bi, g, adj, all_sites, site_idx)
    if center < 0: continue
    base_log = f'$LOGDIR/apt_bond{bi}_c{center}_base.log'
    pert_log = f'$LOGDIR/apt_bond{bi}_c{center}_pert.log'
    if os.path.exists(base_log) and os.path.exists(pert_log): continue
    print(f'{bi} {center}')  # output for bash to process
" 2>/dev/null)

    # Run simulations from bash (so env vars are correct)
    while IFS=' ' read -r BOND_IDX CENTER; do
        [ -z "$BOND_IDX" ] && continue
        BASE_LOG="$LOGDIR/apt_bond${BOND_IDX}_c${CENTER}_base.log"
        PERT_LOG="$LOGDIR/apt_bond${BOND_IDX}_c${CENTER}_pert.log"
        echo "  Bond $BOND_IDX center $CENTER"
        ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep_rect7x7 50 7 1e-3 \
            --center=$CENTER --state=NL > $BASE_LOG 2>&1
        ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 50 7 1e-3 \
            --center=$CENTER --state=NL "--perturb-bond=${BOND_TAG},${J_NEW}" \
            > $PERT_LOG 2>&1
    done <<< "$ASSIGNMENTS"

    # Phase 4: validate
    RESULT=$(python3 -c "
import sys, json
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites
from find_centers import bfs_distances
from bfs_transfer_detect import bfs_transfer_detect, compute_transfer_ratios
from targeted_aperture import find_optimal_center
from searcher import parse_pauli_log

g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}
plan = json.load(open('data/scans/rect_7x7_scan_3gpu.json'))

br = json.load(open('$BROKEN'))
true_bonds = set(br.get('bonds',[br.get('bond',-1)]))

ranked, _ = bfs_transfer_detect('$LOGDIR', plan, g, adj, all_sites, site_idx,
                                  thr=1e-3, ratio_thr=1.05, T_max=7, verbose=False)
candidates = [(bi, score) for bi, score in ranked[:$TOP_K]]

confirmed = []
for bi, score in candidates:
    center, endpoint, cscore = find_optimal_center(bi, g, adj, all_sites, site_idx)
    if center < 0: continue
    base_log = f'$LOGDIR/apt_bond{bi}_c{center}_base.log'
    pert_log = f'$LOGDIR/apt_bond{bi}_c{center}_pert.log'
    try:
        _, _, db = parse_pauli_log(base_log, 'NL')
        _, _, dp = parse_pauli_log(pert_log, 'NL')
    except: continue
    ratios = compute_transfer_ratios(db, dp, [center], g, adj, all_sites, site_idx, 1e-3, T_max=7)
    if bi in ratios:
        best_ratio = max(e[0] for e in ratios[bi])
        if best_ratio > 1.05:
            confirmed.append(bi)

found = set(confirmed) & true_bonds
both = true_bonds.issubset(set(confirmed))
tag = 'BOTH' if both else ('PARTIAL' if found else 'NONE')
print(f'{tag}|{sorted(confirmed)}|{sorted(true_bonds)}')
" 2>/dev/null)

    TAG=$(echo $RESULT | cut -d'|' -f1)
    CONF=$(echo $RESULT | cut -d'|' -f2)
    TRUE=$(echo $RESULT | cut -d'|' -f3)
    total=$((total+1))
    [ "$TAG" = "BOTH" ] && correct=$((correct+1))
    echo "  Result: $TAG  confirmed=$CONF  true=$TRUE"
    echo ""
done

echo "=== FINAL: $correct/$total BOTH ==="
echo "=== DONE $(date) ==="
