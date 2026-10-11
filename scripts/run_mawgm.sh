#!/bin/bash
# run_mawgm.sh — Micro Aperture Wavefront Geometry Measure
#
# 4 scans × 3 cycles, d_max=1, Voronoi ties=0, full 84-bond coverage.
# Baseline+perturbed run in parallel on GPU0+GPU1, scan pairs overlap on GPU2.
#
# Usage:
#   bash scripts/run_mawgm.sh [trial] [seed] [dJ] [n_bonds]

TRIAL=${1:-1}
SEED=${2:-42}
DJ=${3:--0.20}
N_BONDS=${4:-1}

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

GRAPH="data/graphs/rect_7x7.json"
BINARY="./main_pauli_sweep_rect7x7"
SCAN_PLAN="data/scans/rect_7x7_scan_3gpu.json"
CUTOFF=1e-3
CHI=50
STEPS=5   # d_max=2, ceil(2/0.448)=5 cycles
LOGDIR="logs/mawgm_v2/trial_${TRIAL}"
mkdir -p $LOGDIR

echo "=== MAWGM Trial $TRIAL (seed=$SEED, dJ=$DJ, n_bonds=$N_BONDS) ==="
echo "$(date)"

# ── Step 1: Random bond(s) ─────────────────────────────────────────────────────
BOND=$(python3 -c "
import random, json
random.seed($SEED + $TRIAL)
g = json.load(open('$GRAPH'))
bonds = random.sample(range(len(g['edges'])), $N_BONDS)
print(';'.join(map(str, bonds)))
")
BOND_PY="${BOND//;/,}"
J_NEW=$(python3 -c "print(f'{1.0 + ($DJ):.3f}')")

python3 scripts/breaker.py \
    --graph $GRAPH --bond "$BOND_PY" --coupling $DJ \
    --out $LOGDIR/broken.json > /dev/null 2>&1

echo "  True bond(s): $BOND  J_new=$J_NEW"

# ── Step 2: Read scan plan ─────────────────────────────────────────────────────
CENTERS_S1=$(python3 -c "import json; p=json.load(open('$SCAN_PLAN')); print(','.join(map(str,p['placements'][0]['centers'])))")
CENTERS_S2=$(python3 -c "import json; p=json.load(open('$SCAN_PLAN')); print(','.join(map(str,p['placements'][1]['centers'])))")
CENTERS_S3=$(python3 -c "import json; p=json.load(open('$SCAN_PLAN')); print(','.join(map(str,p['placements'][2]['centers'])))")
# ── Step 3: Run 3 scans in parallel — one GPU per scan, baseline then perturbed ──
run_scan() {
    local S=$1; local CSTR=$2; local GPU=$3
    local PERTURB_ARG="--perturb-bond=${BOND},${J_NEW}"
    cd /scratch/MPF/CppSim
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$CSTR --state=NL \
        > $LOGDIR/scan${S}_base.log 2>&1
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$CSTR --state=NL "$PERTURB_ARG" \
        > $LOGDIR/scan${S}_pert.log 2>&1
}

echo "  Running 3 scans in parallel — one GPU per scan..."
# GPU0: scan1 (corners+center), GPU1: scan2 (quadrants), GPU2: scan3 (mid-edges)
# Each GPU: baseline then perturbed sequentially — wall-clock = 1 scan time
( run_scan 1 $CENTERS_S1 0 ) &
( run_scan 2 $CENTERS_S2 1 ) &
( run_scan 3 $CENTERS_S3 2 ) &
wait
echo "  All scans done."

# ── Step 4: Winner-takes-all across all scans ──────────────────────────────────
python3 - <<PYEOF
import sys, numpy as np, json
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
from find_centers import bond_distances

FLOOR=1e-7; thr=1e-3; T_CLEAN=3  # thr >= cutoff=1e-3; T_CLEAN=3 avoids secondary wave inversion
g = load_graph('$GRAPH')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}

plan     = json.load(open('$SCAN_PLAN'))
br       = json.load(open('$LOGDIR/broken.json'))
true_bonds = set(br.get('bonds', [br.get('bond',-1)]))

best_contrast=-1; best_bond=-1; best_scan=-1; best_center=-1

for scan_info in plan['placements']:
    sid     = scan_info['scan']
    centers = scan_info['centers']
    try:
        _, _, db = parse_pauli_log(f'$LOGDIR/scan{sid}_base.log', 'NL')
        _, _, dp = parse_pauli_log(f'$LOGDIR/scan{sid}_pert.log', 'NL')
    except Exception as e:
        print(f'  Scan {sid}: ERROR {e}'); continue

    nc = min(db.shape[0], dp.shape[0], T_CLEAN)
    ns = db.shape[1]
    L_pos = np.zeros(ns); L_neg = np.zeros(ns)
    for v in range(ns):
        for t in range(nc):
            if db[t,v] > thr:
                val = -np.log(max(dp[t,v],FLOOR)/db[t,v])
                if val > L_pos[v]: L_pos[v] = val
                if val < L_neg[v]: L_neg[v] = val
    L_signed = np.where(np.abs(L_pos)>=np.abs(L_neg), L_pos, L_neg)

    site_bond_dist = {c: bond_distances(g, c) for c in centers}
    for bi in scan_info['bonds_covered']:
        e = g['edges'][bi]
        a, b = get_bond_sites(e)
        contrast = abs(L_signed[site_idx[a]] - L_signed[site_idx[b]])
        if contrast > best_contrast:
            best_contrast = contrast; best_bond = bi; best_scan = sid
            best_center = min(centers, key=lambda c: site_bond_dist[c][bi])

correct = best_bond in true_bonds
ea,eb = get_bond_sites(g['edges'][best_bond]) if best_bond>=0 else (0,0)
print(f'  Declared: Bond {best_bond} ({ea}↔{eb})  scan={best_scan}  contrast={best_contrast:.4f}')
print(f'  Ground truth: {sorted(true_bonds)}')
print(f'  Result: {"✓ CORRECT" if correct else "✗ WRONG"}')

result = {'trial': $TRIAL, 'bonds': '$BOND', 'n_bonds': $N_BONDS,
          'true_bonds': sorted(true_bonds),
          'declared_bond': best_bond, 'declared_sites': [ea,eb],
          'winning_scan': best_scan, 'winning_center': best_center,
          'contrast': best_contrast, 'correct': correct}
with open('$LOGDIR/result.json','w') as f: json.dump(result,f,indent=2)
print(f'  Saved: $LOGDIR/result.json')
PYEOF

echo "=== Trial $TRIAL DONE $(date) ==="
