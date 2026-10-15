#!/bin/bash
# run_all_trials_aperture.sh — Full pipeline: candidates + aperture pool + validation
#
# Usage: bash scripts/run_all_trials_aperture.sh

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate 2>/dev/null || true

LOGBASE="logs/mawgm_f64_2b"
correct=0; total=0

echo "=== Full aperture pipeline: all 10 trials ==="
echo "$(date)"
echo ""

for TRIAL in $(seq 1 10); do
    LOGDIR="${LOGBASE}/trial_${TRIAL}"
    BROKEN="${LOGDIR}/broken.json"
    [ ! -f "$BROKEN" ] && continue

    BOND_TAG=$(python3 -c "
import json; br=json.load(open('$BROKEN'))
print(';'.join(map(str,sorted(br.get('bonds',[br.get('bond',-1)])))))")
    J_NEW=$(python3 -c "
import json; br=json.load(open('$BROKEN'))
print(f\"{br.get('perturbation',{}).get('J_perturbed',0.8):.3f}\")")

    echo "--- Trial $TRIAL: bonds=$BOND_TAG J_new=$J_NEW ---"

    # Get candidates with centers ON near endpoint
    CANDIDATES=$(python3 -c "
import sys, json
sys.path.insert(0,'scripts')
from searcher import load_graph, build_adj, get_bond_sites
from bfs_detect import bfs_detect

g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}
plan = json.load(open('data/scans/rect_7x7_scan_3gpu.json'))

# Use BFS sign-flip method — finds both true bonds in 9/10 trials
ranked, _ = bfs_detect('$LOGDIR', plan, g, adj, all_sites, site_idx,
                         v_hop=0.448, thr=1e-3, thr_L=0.05, T_max=5, verbose=False)
pairs = []
for bi, score in ranked[:5]:
    a, b = get_bond_sites(g['edges'][bi])
    # Center ON near endpoint — J=0.8 transmits enough for far endpoint to cross thr
    row_a=(a-1)//7; row_b=(b-1)//7
    center = a if row_a <= row_b else b
    pairs.append(f'{bi}:{center}')
print(' '.join(pairs))
" 2>/dev/null)

    [ -z "$CANDIDATES" ] && echo "  No candidates found" && continue
    echo "  Candidates: $CANDIDATES"

    # Run aperture pool
    bash scripts/run_aperture_pool.sh $LOGDIR "$BOND_TAG" "$J_NEW" "$CANDIDATES" 2>/dev/null

    # Validate
    RESULT=$(python3 scripts/validate_aperture.py \
        --logdir $LOGDIR \
        --candidates "$CANDIDATES" \
        --broken $BROKEN \
        --T_max 5 --thr_L 0.05 2>/dev/null | tail -3)

    echo "$RESULT"
    echo "$RESULT" | grep -q "BOTH" && correct=$((correct+1))
    total=$((total+1))
    echo ""
done

echo "=== FINAL: $correct/$total BOTH ==="
echo "=== DONE $(date) ==="
