#!/usr/bin/env python3
"""
vote_bonds.py — Declaration voting across all per-center results.

Collects all per-center bond declarations, shows which true bonds
were found and by which centers. Works for single and multi-bond faults.

Usage:
  python3 scripts/vote_bonds.py \
      --results  logs/tomo_6c/result_c*.json \
      --broken   logs/tomo_6c/broken.json \
      --graph    data/graphs/rect_7x7.json

  # Or with glob pattern:
  python3 scripts/vote_bonds.py \
      --results "logs/tomo_6c/result_c*.json" \
      --broken  logs/tomo_6c/broken.json \
      --graph   data/graphs/rect_7x7.json
"""

import argparse
import json
import glob
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, get_bond_sites

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='Glob pattern or list of result JSONs')
    ap.add_argument('--broken',  required=True, help='breaker.py JSON (ground truth)')
    ap.add_argument('--graph',   required=True, help='Graph JSON')
    args = ap.parse_args()

    result_files = sorted(glob.glob(args.results))
    if not result_files:
        print(f"ERROR: no files matching {args.results}")
        return

    g = load_graph(args.graph)
    with open(args.broken) as f:
        broken = json.load(f)

    true_bonds = set(broken.get('bonds', [broken.get('bond', -1)]))
    true_bonds.discard(-1)

    print(f"\n=== Declaration Voting ===")
    print(f"  Ground truth: Bonds {sorted(true_bonds)}")
    print(f"  Result files: {len(result_files)}")
    print()

    # Collect declarations
    declarations = {}   # bond_idx -> list of center names
    for rf in result_files:
        with open(rf) as f:
            r = json.load(f)
        center = rf.replace('logs/tomo_6c/result_c', '').replace('.json', '')
        b = r.get('predicted_bond', -1)
        if b >= 0:
            if b not in declarations:
                declarations[b] = []
            declarations[b].append(center)

    # Print ranked by votes
    print("  Bond declarations (ranked by votes):")
    for b, centers in sorted(declarations.items(), key=lambda x: -len(x[1])):
        a, sb = get_bond_sites(g['edges'][b])
        tag = '<-- TRUE BOND' if b in true_bonds else ''
        print(f"    Bond {b:3d} ({a:3d}↔{sb:3d}): "
              f"{len(centers)} vote(s) from centers [{','.join(centers)}] {tag}")

    # Voting result
    found = set(declarations.keys()) & true_bonds
    missing = true_bonds - found
    false_pos = set(declarations.keys()) - true_bonds

    print()
    print(f"  True bonds found:   {sorted(found)}")
    if missing:
        print(f"  True bonds MISSED:  {sorted(missing)}")
    if false_pos:
        voted = {b: len(declarations[b]) for b in false_pos}
        print(f"  False positives:    {sorted(false_pos)} (votes: {voted})")

    if found == true_bonds:
        print(f"\n  Result: ✓ ALL TRUE BONDS FOUND by voting")
    elif found:
        print(f"\n  Result: ~ PARTIAL ({len(found)}/{len(true_bonds)} bonds found)")
    else:
        print(f"\n  Result: ✗ NO TRUE BONDS FOUND")

if __name__ == '__main__':
    main()
