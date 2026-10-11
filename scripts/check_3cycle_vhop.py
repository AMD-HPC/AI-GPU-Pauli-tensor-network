import sys, numpy as np
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj

g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}

_, _, db = parse_pauli_log('/tmp/apt_c4_3cycle_vhop.log', 'NL')

print("Center 4, 3 cycles, v_hop=0.448")
print("Expected: site 7 exactly 0 for all cycles (3 hops > 1.34 lightcone radius)")
print()
print(f"{'Cycle':>5}  {'site':>5}  {'n(v,t)':>12}  {'above 1e-4?':>12}")
print("-"*50)
for t in range(3):
    for s in [4, 5, 6, 7]:
        nv = db[t, site_idx[s]]
        flag = 'YES ***' if nv > 1e-4 else 'no'
        print(f"{t+1:>5}  {s:>5}  {nv:>12.7f}  {flag:>12}")
    print()
