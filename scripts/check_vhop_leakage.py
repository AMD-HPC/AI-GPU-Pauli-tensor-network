import sys, numpy as np
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj

g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
site_idx = {s: i for i, s in enumerate(all_sites)}

_, _, db_vhop   = parse_pauli_log('/tmp/apt_c4_vhop.log',   'NL')
_, _, db_novhop = parse_pauli_log('/tmp/apt_c4_novhop.log', 'NL')

print(f"{'Cycle':>5}  {'site':>5}  {'n(v_hop)':>12}  {'n(no_hop)':>12}  {'diff':>12}")
print("-"*60)
for t in range(5):
    for s in [4, 5, 6, 7]:
        nv = db_vhop[t, site_idx[s]]
        nn = db_novhop[t, site_idx[s]]
        flag = ' *** LR!' if s==7 and t==0 and nn>1e-4 else ''
        print(f"{t+1:>5}  {s:>5}  {nv:>12.7f}  {nn:>12.7f}  {nn-nv:>12.7f}{flag}")
    print()
