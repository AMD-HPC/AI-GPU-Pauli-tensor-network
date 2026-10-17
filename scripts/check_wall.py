import sys, numpy as np
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj

FLOOR=1e-7; thr=1e-3; thr_L=0.05
g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}

_, _, db = parse_pauli_log('logs/mawgm_f64_2b/trial_3/wall_b34_base.log','NL')
_, _, dp = parse_pauli_log('logs/mawgm_f64_2b/trial_3/wall_b34_pert.log','NL')

a, b = 40, 41
nc = min(db.shape[0], dp.shape[0])
best_La=0; best_Lb=0; best_cyc=-1

print(f"Wall antenna validation — bond 34 ({a}<>{b})")
print(f"{'Cycle':>5}  {'L(40)':>8}  {'L(41)':>8}  {'pattern'}")
print("-"*45)

for t in range(nc):
    nb_a=db[t,site_idx[a]]; np_a=dp[t,site_idx[a]]
    nb_b=db[t,site_idx[b]]; np_b=dp[t,site_idx[b]]
    La=-np.log(max(np_a,FLOOR)/nb_a) if nb_a>thr else 0.0
    Lb=-np.log(max(np_b,FLOOR)/nb_b) if nb_b>thr else 0.0
    flip = ''
    if (La<-thr_L and Lb>thr_L):
        flip = 'BLUE-RED ✓'
    elif (La>thr_L and Lb<-thr_L):
        flip = 'RED-BLUE ✓'
    print(f"  {t+1:>3}  {La:>+8.3f}  {Lb:>+8.3f}  {flip}")
    if flip and abs(La)+abs(Lb) > abs(best_La)+abs(best_Lb):
        best_La=La; best_Lb=Lb; best_cyc=t+1

print()
if best_cyc > 0:
    print(f"Bond 34: CONFIRMED at cycle {best_cyc}  L(40)={best_La:+.3f}  L(41)={best_Lb:+.3f}")
else:
    print("Bond 34: REJECTED — no sign flip")
