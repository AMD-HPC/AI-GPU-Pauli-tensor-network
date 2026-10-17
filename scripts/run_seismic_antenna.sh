#!/bin/bash
# run_seismic_antenna.sh — Complete seismic + antenna pipeline for all 10 trials
#
# Phase 1: BFS candidates from existing 3-scan seismic logs
# Phase 2: Wall antenna per candidate (column/row of centers, one side)
# Phase 3: Sign flip validation
#
# Usage: bash scripts/run_seismic_antenna.sh

cd /scratch/MPF/CppSim
source /scratch/.venv/bin/activate 2>/dev/null || true

LOGBASE="logs/mawgm_f64_2b"
BINARY="./main_pauli_sweep_rect7x7"
CHI=50; STEPS=5; CUTOFF=1e-3

correct=0; total=0

echo "=== Seismic + Antenna Pipeline: all 10 trials ==="
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

    # Phase 1: get BFS candidates
    # High confidence (score > CONF_THR): confirmed directly from seismic, no antenna
    # Low confidence (score <= CONF_THR): uncertain, run wall antenna
    CONF_THR=1.5

    ASSIGNMENTS=$(python3 -c "
import sys, json
sys.path.insert(0,'scripts')
from searcher import load_graph, build_adj, get_bond_sites
from bfs_detect import bfs_detect
from wall_aperture import find_wall_centers

g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}
plan = json.load(open('data/scans/rect_7x7_scan_3gpu.json'))
CONF_THR = $CONF_THR

ranked, _ = bfs_detect('$LOGDIR', plan, g, adj, all_sites, site_idx,
                         v_hop=0.448, thr=1e-3, thr_L=0.05, T_max=5, verbose=False)

confirmed_direct = []
for bi, score in ranked[:5]:
    a,b = get_bond_sites(g['edges'][bi])
    if score > CONF_THR:
        # High confidence: confirmed directly from seismic
        print(f'{bi}|DIRECT|{a}|score={score:.2f}')
    else:
        # Low confidence: need wall antenna
        wall, orient, direction, near, a, b = find_wall_centers(bi, g, set(all_sites))
        if not wall: continue
        cstr = ','.join(map(str, wall))
        print(f'{bi}|ANTENNA|{cstr}|{near}')
" 2>/dev/null)

    [ -z "$ASSIGNMENTS" ] && echo "  No candidates" && continue

    # Phase 2+3: process each candidate
    confirmed=()
    while IFS='|' read -r BOND_IDX METHOD REST; do
        [ -z "$BOND_IDX" ] && continue

        if [ "$METHOD" = "DIRECT" ]; then
            # High confidence — confirmed from seismic, no antenna needed
            SCORE=$(echo $REST | grep -oP 'score=\K[0-9.]+')
            echo "  Bond $BOND_IDX: CONFIRMED (seismic score=$SCORE, no antenna needed)"
            confirmed+=($BOND_IDX)

        elif [ "$METHOD" = "ANTENNA" ]; then
            # Low confidence — queue for GPU pool
            CENTERS=$(echo $REST | cut -d'|' -f1)
            NEAR=$(echo $REST | cut -d'|' -f2)
            ANTENNA_QUEUE+=("${BOND_IDX}|${CENTERS}|${NEAR}")
        fi
    done <<< "$ASSIGNMENTS"

    # Run antenna queue with GPU pool (3 GPUs, base+pert sequential on same GPU)
    run_antenna_gpu() {
        local BOND_IDX=$1; local CENTERS=$2; local GPU=$3
        local BASE_LOG="$LOGDIR/wall_b${BOND_IDX}_base.log"
        local PERT_LOG="$LOGDIR/wall_b${BOND_IDX}_pert.log"
        [ -f "$BASE_LOG" ] && return  # already exists
        ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
            --center=$CENTERS --state=NL > $BASE_LOG 2>&1
        ROCR_VISIBLE_DEVICES=$GPU $BINARY $CHI $STEPS $CUTOFF \
            --center=$CENTERS --state=NL "--perturb-bond=${BOND_TAG},${J_NEW}" \
            > $PERT_LOG 2>&1
    }

    GPU=0
    PIDS=()
    for ENTRY in "${ANTENNA_QUEUE[@]}"; do
        BI=$(echo $ENTRY | cut -d'|' -f1)
        CS=$(echo $ENTRY | cut -d'|' -f2)
        run_antenna_gpu $BI "$CS" $GPU &
        PIDS+=($!)
        GPU=$(( (GPU+1) % 3 ))
        if [ $GPU -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
    done
    [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"

    # Validate all antenna results
    confirmed=("${DIRECT_CONFIRMED[@]}")
    for ENTRY in "${ANTENNA_QUEUE[@]}"; do
        BOND_IDX=$(echo $ENTRY | cut -d'|' -f1)
        BASE_LOG="$LOGDIR/wall_b${BOND_IDX}_base.log"
        PERT_LOG="$LOGDIR/wall_b${BOND_IDX}_pert.log"
        if [ ! -f "$BASE_LOG" ]; then
            echo "  Bond $BOND_IDX: ERROR (no log)"
            continue
        fi
        RESULT=$(python3 -c "
import sys, numpy as np
sys.path.insert(0,'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
FLOOR=1e-7; thr=1e-3; thr_L=0.05
g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}
try:
    _, _, db = parse_pauli_log('$BASE_LOG','NL')
    _, _, dp = parse_pauli_log('$PERT_LOG','NL')
except: print('ERROR|0|0'); exit()
a,b = get_bond_sites(g['edges'][$BOND_IDX])
nc = min(db.shape[0], dp.shape[0])
best=0.0; best_cyc=-1
for t in range(nc):
    nb_a=db[t,site_idx[a]]; np_a=dp[t,site_idx[a]]
    nb_b=db[t,site_idx[b]]; np_b=dp[t,site_idx[b]]
    La=-np.log(max(np_a,FLOOR)/nb_a) if nb_a>thr else 0.0
    Lb=-np.log(max(np_b,FLOOR)/nb_b) if nb_b>thr else 0.0
    if (La<-thr_L and Lb>thr_L) or (La>thr_L and Lb<-thr_L):
        contrast=abs(La)+abs(Lb)
        if contrast>best: best=contrast; best_cyc=t+1
if best_cyc>0: print(f'CONFIRMED|{best_cyc}|{best:.3f}')
else: print('REJECTED|0|0')
" 2>/dev/null)
        TAG=$(echo $RESULT | cut -d'|' -f1)
        CYC=$(echo $RESULT | cut -d'|' -f2)
        CONT=$(echo $RESULT | cut -d'|' -f3)
        echo "  Bond $BOND_IDX: $TAG [antenna] (cycle=$CYC contrast=$CONT)"
        [ "$TAG" = "CONFIRMED" ] && confirmed+=($BOND_IDX)
        if [ "$METHOD" = "DUMMY" ]; then  # unreachable, keeps syntax valid

            # Validate sign flip
            RESULT=$(python3 -c "
import sys, numpy as np
sys.path.insert(0,'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
FLOOR=1e-7; thr=1e-3; thr_L=0.05
g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}
try:
    _, _, db = parse_pauli_log('$BASE_LOG','NL')
    _, _, dp = parse_pauli_log('$PERT_LOG','NL')
except: print('ERROR|0|0'); exit()
a,b = get_bond_sites(g['edges'][$BOND_IDX])
nc = min(db.shape[0], dp.shape[0])
best=0.0; best_cyc=-1
for t in range(nc):
    nb_a=db[t,site_idx[a]]; np_a=dp[t,site_idx[a]]
    nb_b=db[t,site_idx[b]]; np_b=dp[t,site_idx[b]]
    La=-np.log(max(np_a,FLOOR)/nb_a) if nb_a>thr else 0.0
    Lb=-np.log(max(np_b,FLOOR)/nb_b) if nb_b>thr else 0.0
    if (La<-thr_L and Lb>thr_L) or (La>thr_L and Lb<-thr_L):
        contrast=abs(La)+abs(Lb)
        if contrast>best: best=contrast; best_cyc=t+1
if best_cyc>0: print(f'CONFIRMED|{best_cyc}|{best:.3f}')
else: print('REJECTED|0|0')
" 2>/dev/null)

            TAG=$(echo $RESULT | cut -d'|' -f1)
            CYC=$(echo $RESULT | cut -d'|' -f2)
            CONT=$(echo $RESULT | cut -d'|' -f3)
            echo "  Bond $BOND_IDX: $TAG [antenna] (cycle=$CYC contrast=$CONT)"
            [ "$TAG" = "CONFIRMED" ] && confirmed+=($BOND_IDX)
        fi
    done <<< "$ASSIGNMENTS"

    # Check result
    python3 -c "
import json
br=json.load(open('$BROKEN'))
true_bonds=set(br.get('bonds',[br.get('bond',-1)]))
confirmed=set([$(IFS=,; echo "${confirmed[*]}" | sed 's/ /,/g')])
both=true_bonds.issubset(confirmed)
found=true_bonds&confirmed
tag='BOTH' if both else ('PARTIAL' if found else 'NONE')
print(f'  Result: {tag}  confirmed={sorted(confirmed)}  true={sorted(true_bonds)}')
" 2>/dev/null

    total=$((total+1))
    echo ""
done

echo "=== DONE $(date) ==="
