#!/bin/bash
# run_adaptive_tomo.sh — Adaptive seismic → aperture tomography
#
# Protocol:
#   1. Random bond selection (seeded per trial)
#   2. Seismic run: all 7 Voronoi centers simultaneous → max-contrast bond detection
#   3. If seismic correct → DONE (skip aperture)
#   4. If seismic wrong  → Aperture: 7 individual center runs → ensemble detection
#
# Voronoi centers (ties=0, max_T_b=5, all 84 bonds covered in 5 cycles):
#   sites 3,13,15,25,35,37,47 — no two share row or column
#
# Usage:
#   bash scripts/run_adaptive_tomo.sh [trial_id] [seed] [dJ]
#   bash scripts/run_adaptive_tomo.sh 1 42 -0.20

TRIAL=${1:-1}
SEED=${2:-42}
DJ=${3:--0.20}

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate

GRAPH="data/graphs/rect_7x7.json"
BINARY="./main_pauli_sweep_rect7x7"
CUTOFF=1e-3
CHI=50
STEPS=5
VORONOI_CENTERS="3,13,15,25,35,37,47"
N_BONDS=${5:-1}   # number of bonds to break (default 1)
LOGDIR="logs/adaptive_tomo_${N_BONDS}b/trial_${TRIAL}"
mkdir -p $LOGDIR

echo "================================================================"
echo "=== Adaptive Tomography Trial $TRIAL (seed=$SEED, dJ=$DJ, n_bonds=$N_BONDS) ==="
echo "================================================================"
echo "$(date)"

# ── Step 1: Random bond(s) ─────────────────────────────────────────────────────
echo ""
echo "--- Step 1: Random bond selection (n=$N_BONDS) ---"
BOND=$(python3 -c "
import random, json
random.seed($SEED + $TRIAL)
g = json.load(open('$GRAPH'))
n = len(g['edges'])
bonds = random.sample(range(n), $N_BONDS)
print(';'.join(map(str,bonds)))
")
BOND_PY="${BOND//;/,}"
J_NEW=$(python3 -c "print(f'{1.0 + ($DJ):.3f}')")

python3 scripts/breaker.py \
    --graph $GRAPH --bond "$BOND_PY" --coupling $DJ \
    --out $LOGDIR/broken.json > /dev/null 2>&1

BOND_INFO=$(python3 -c "
import json
b = json.load(open('$LOGDIR/broken.json'))
print(f\"bond {b['bond']} ({b['site_a']}↔{b['site_b']})\")
")
echo "  True: $BOND_INFO  J_new=$J_NEW"

# ── Step 2: Seismic ────────────────────────────────────────────────────────────
echo ""
echo "--- Step 2: Seismic (7 Voronoi centers simultaneous) ---"

ROCR_VISIBLE_DEVICES=0 $BINARY $CHI $STEPS $CUTOFF \
    --center=$VORONOI_CENTERS --state=NL \
    > $LOGDIR/seismic_base.log 2>&1 &
PID_BASE=$!

ROCR_VISIBLE_DEVICES=1 $BINARY $CHI $STEPS $CUTOFF \
    --center=$VORONOI_CENTERS --state=NL \
    "--perturb-bond=${BOND},${J_NEW}" \
    > $LOGDIR/seismic_pert.log 2>&1 &
PID_PERT=$!

wait $PID_BASE $PID_PERT
echo "  Seismic done."

# Detect bond from seismic using global max-contrast in clean window
SEISMIC_OUT=$(python3 -c "
import sys, numpy as np, json
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

FLOOR=1e-7; thr=1e-4; T_CLEAN=3
_, _, db = parse_pauli_log('$LOGDIR/seismic_base.log', 'NL')
_, _, dp = parse_pauli_log('$LOGDIR/seismic_pert.log', 'NL')
nc = min(db.shape[0], dp.shape[0], T_CLEAN)
ns = db.shape[1]
g = load_graph('$GRAPH')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}

L = np.zeros(ns)
for v in range(ns):
    for t in range(nc):
        if db[t,v] > thr:
            val = -np.log(max(dp[t,v],FLOOR)/db[t,v])
            if val > L[v]: L[v] = val

best_bond=-1; best_contrast=-1
for bi, e in enumerate(g['edges']):
    a,b = get_bond_sites(e)
    c = abs(L[site_idx[a]] - L[site_idx[b]])
    if c > best_contrast: best_contrast=c; best_bond=bi

br = json.load(open('$LOGDIR/broken.json'))
true_bonds = set(br.get('bonds',[br.get('bond',-1)]))
correct = best_bond in true_bonds
ea,eb = get_bond_sites(g['edges'][best_bond]) if best_bond>=0 else (0,0)
print(f'{best_bond}|{ea}|{eb}|{correct}|{best_contrast:.4f}')
" 2>/dev/null)

SEISMIC_BOND=$(echo $SEISMIC_OUT | cut -d'|' -f1)
SEISMIC_A=$(echo $SEISMIC_OUT | cut -d'|' -f2)
SEISMIC_B=$(echo $SEISMIC_OUT | cut -d'|' -f3)
SEISMIC_CORRECT=$(echo $SEISMIC_OUT | cut -d'|' -f4)
SEISMIC_CONTRAST=$(echo $SEISMIC_OUT | cut -d'|' -f5)

echo "  Seismic declares: Bond $SEISMIC_BOND ($SEISMIC_A↔$SEISMIC_B)  contrast=$SEISMIC_CONTRAST"

if [ "$SEISMIC_CORRECT" = "True" ] && [ "$N_BONDS" = "1" ]; then
    echo "  ✓ Seismic CORRECT — skipping aperture"
    python3 -c "
import json
br = json.load(open('$LOGDIR/broken.json'))
true_bonds = br.get('bonds', [br.get('bond',-1)])
result = {'trial': $TRIAL, 'bonds': '$BOND', 'n_bonds': $N_BONDS,
          'true_bonds': true_bonds,
          'true_a': br['site_a'], 'true_b': br['site_b'],
          'seismic_bond': $SEISMIC_BOND, 'seismic_correct': True,
          'aperture_run': False, 'aperture_correct': None,
          'final_correct': True, 'method': 'seismic'}
with open('$LOGDIR/result.json','w') as f: json.dump(result,f,indent=2)
print(f'  Saved: $LOGDIR/result.json')
" 2>/dev/null
    echo ""
    echo "=== Trial $TRIAL: ✓ CORRECT (seismic) $(date) ==="
    exit 0
fi

if [ "$SEISMIC_CORRECT" = "True" ]; then
    echo "  ✓ Seismic found one bond — running aperture for remaining..."
else
    echo "  ✗ Seismic WRONG — running aperture..."
fi

# ── Step 3: Aperture (same 7 Voronoi centers, independent runs) ───────────────
echo ""
echo "--- Step 3: Aperture (7 centers, 3 GPUs) ---"

run_center() {
    local C=$1; local GPU=$2
    cd /scratch/MPF/CppSim
    local PERTURB_ARG="--perturb-bond=${BOND},${J_NEW}"
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C --state=NL \
        > $LOGDIR/apt_base_c${C}.log 2>&1
    ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
        --center=$C --state=NL "$PERTURB_ARG" \
        > $LOGDIR/apt_pert_c${C}.log 2>&1
    echo "  [GPU$GPU] Center $C done."
}

( run_center 3  0; run_center 13 0; run_center 15 0 ) &
( run_center 25 1; run_center 35 1 ) &
( run_center 37 2; run_center 47 2 ) &
wait
echo "  Aperture done."

# Detect bond from aperture: winner-takes-all across independent center runs
# The center with highest signed contrast across all its bonds wins — no summation.
APT_OUT=$(python3 -c "
import sys, numpy as np, json
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

FLOOR=1e-7; thr=1e-4; T_CLEAN=5
g = load_graph('$GRAPH')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}
centers = [int(c) for c in '$VORONOI_CENTERS'.split(',')]

best_contrast=-1; best_bond=-1; best_center=-1
for c in centers:
    try:
        _, _, db = parse_pauli_log('$LOGDIR/apt_base_c'+str(c)+'.log', 'NL')
        _, _, dp = parse_pauli_log('$LOGDIR/apt_pert_c'+str(c)+'.log', 'NL')
    except: continue
    nc = min(db.shape[0], dp.shape[0], T_CLEAN)
    L_pos = np.zeros(len(all_sites)); L_neg = np.zeros(len(all_sites))
    for v in range(len(all_sites)):
        for t in range(nc):
            if db[t,v] > thr:
                val = -np.log(max(dp[t,v],FLOOR)/db[t,v])
                if val > L_pos[v]: L_pos[v] = val
                if val < L_neg[v]: L_neg[v] = val
    L_signed = np.where(np.abs(L_pos)>=np.abs(L_neg), L_pos, L_neg)
    for bi, e in enumerate(g['edges']):
        a, b = get_bond_sites(e)
        contrast = abs(L_signed[site_idx[a]] - L_signed[site_idx[b]])
        if contrast > best_contrast:
            best_contrast=contrast; best_bond=bi; best_center=c

br = json.load(open('$LOGDIR/broken.json'))
true_bonds = set(br.get('bonds',[br.get('bond',-1)]))
correct = best_bond in true_bonds
ea,eb = get_bond_sites(g['edges'][best_bond]) if best_bond>=0 else (0,0)
print(f'{best_bond}|{ea}|{eb}|{correct}|{best_center}|{best_contrast:.4f}')
" 2>/dev/null)

APT_BOND=$(echo $APT_OUT | cut -d'|' -f1)
APT_A=$(echo $APT_OUT | cut -d'|' -f2)
APT_B=$(echo $APT_OUT | cut -d'|' -f3)
APT_CORRECT_STR=$(echo $APT_OUT | cut -d'|' -f4)
APT_WINNER=$(echo $APT_OUT | cut -d'|' -f5)
APT_CONTRAST=$(echo $APT_OUT | cut -d'|' -f6)

echo "  Aperture winner: center $APT_WINNER → Bond $APT_BOND ($APT_A↔$APT_B)  contrast=$APT_CONTRAST"

FINAL_CORRECT="False"
[ "$APT_CORRECT_STR" = "True" ] && FINAL_CORRECT="True"

python3 -c "
import json
br = json.load(open('$LOGDIR/broken.json'))
true_bonds = br.get('bonds', [br.get('bond',-1)])
found_bonds = list(set(filter(lambda x: x>=0, [$SEISMIC_BOND if '$SEISMIC_CORRECT'=='True' else -1, $APT_BOND])))
final_correct = all(b in found_bonds for b in true_bonds) if $N_BONDS==1 else len(set(found_bonds) & set(true_bonds)) > 0
result = {'trial': $TRIAL, 'bonds': '$BOND', 'n_bonds': $N_BONDS,
          'true_bonds': true_bonds,
          'true_a': br['site_a'], 'true_b': br['site_b'],
          'seismic_bond': $SEISMIC_BOND, 'seismic_correct': '$SEISMIC_CORRECT'=='True',
          'aperture_run': True, 'aperture_bond': $APT_BOND,
          'aperture_winner_center': $APT_WINNER,
          'aperture_correct': '$FINAL_CORRECT'=='True',
          'found_bonds': found_bonds,
          'final_correct': final_correct,
          'method': 'aperture'}
with open('$LOGDIR/result.json','w') as f: json.dump(result,f,indent=2)
" 2>/dev/null

echo ""
if [ "$FINAL_CORRECT" = "True" ]; then
    echo "=== Trial $TRIAL: ✓ CORRECT (aperture) $(date) ==="
else
    echo "=== Trial $TRIAL: ✗ FAIL $(date) ==="
fi
