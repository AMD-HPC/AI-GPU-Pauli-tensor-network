#!/usr/bin/env python3
"""
shots_needed.py — Detection rate vs N echo-circuit shots (OTOC shot noise).

n(v,t) on hardware is an OTOC measured via the echo protocol (Google quantum
echoes family). This script models the shot noise by injecting binomial sampling
into simulated n_pert(v,t) from existing logs, then reruns bfs_detect on the
noisy estimate n̂_pert with n_base exact (from simulation).

Noise model: F_sigma(v,t) = 1 - (4/3)*n(v,t)  [isotropic: X=Y=Z]
  Each sigma: N shots of ±1, E[outcome]=F_sigma
  n̂ = (3 - F̂_X - F̂_Y - F̂_Z) / 4
  Total shots per site per cycle = 3*N (one N per sigma).

Also computes false-positive rate by running detection on a healthy device
(n_pert_true = n_base, only shot noise).

Usage (from CppSim root, inside venv):
    python3 -I scripts/shots_needed.py \\
        --logbase  logs/mawgm_f64 \\
        --trials   1 2 3 4 5 6 7 8 9 10 \\
        --plan     data/scans/rect_7x7_scan_3gpu.json \\
        --graph    data/graphs/rect_7x7.json \\
        --N_list   1000 10000 100000 1000000 \\
        --K        100 \\
        --seed     42 \\
        --out      logs/shots_needed_results.json \\
        --plot     logs/shots_needed.png
"""

import argparse
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
from bfs_detect import (threshold_crossing, bfs_with_color, bfs_detect_scan,
                        is_collision, bond_transfer_L, FLOOR)


# ---------------------------------------------------------------------------
# Shot noise injection
# ---------------------------------------------------------------------------

def inject_noise_otoc(n_true, N_per_sigma, rng):
    """
    OTOC echo shot noise model.
    F_sigma = 1 - (4/3)*n  (isotropic X=Y=Z assumption)
    N_per_sigma shots of ±1 per sigma butterfly.
    Total shots per site = 3 * N_per_sigma.
    """
    n = np.clip(n_true, 0.0, 1.0)
    F = 1.0 - (4.0 / 3.0) * n          # F in [-1/3, 1]
    p_plus = np.clip((1.0 + F) / 2.0, 0.0, 1.0)

    # Sample X, Y, Z butterflies independently
    F_hats = []
    for _ in range(3):
        k = rng.binomial(N_per_sigma, p_plus)
        F_hats.append((2.0 * k - N_per_sigma) / N_per_sigma)

    n_hat = (3.0 - F_hats[0] - F_hats[1] - F_hats[2]) / 4.0
    return np.clip(n_hat, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Multi-scan detection on pre-loaded numpy arrays
# ---------------------------------------------------------------------------

def detect_arrays(scan_data, plan, g, adj, all_sites, site_idx,
                  v_hop=0.448, thr=1e-3, thr_L=0.05, T_max=5):
    """
    Run multi-scan bfs_detect on pre-loaded numpy arrays.

    scan_data: dict  scan_id -> {'base': ndarray, 'pert': ndarray}
                     arrays shape = (n_cycles, n_sites)

    Returns ranked list [(bond_idx, score), ...].
    """
    bond_score = {}

    for sc in plan['placements']:
        sid = sc['scan']
        centers = sc['centers']
        all_bonds = set(range(len(g['edges'])))

        entry = scan_data.get(sid)
        if entry is None:
            continue
        db = entry['base']
        dp = entry['pert']

        cands = bfs_detect_scan(db, dp, centers, all_bonds, g, adj,
                                all_sites, site_idx, v_hop,
                                thr, thr_L, T_max, verbose=False)

        for c in cands:
            bi = c['bond']
            weight = abs(c['L']) / c['t_base']
            if c['delayed']:       weight *= 2.0
            if c['is_collision']:  weight *= 0.3
            bond_score[bi] = bond_score.get(bi, 0.0) + weight

    ranked = sorted(bond_score.items(), key=lambda x: -x[1])
    return ranked


def rank_of_bond(ranked, true_bond):
    """Return 1-based rank of true_bond in ranked list, or n_bonds+1 if absent."""
    for i, (bi, _) in enumerate(ranked):
        if bi == true_bond:
            return i + 1
    return len(ranked) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logbase',  default='logs/mawgm_f64',
                    help='Base directory containing trial_N subdirs')
    ap.add_argument('--trials',   nargs='+', type=int,
                    default=list(range(1, 11)),
                    help='Trial numbers to include')
    ap.add_argument('--plan',     default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',    default='data/graphs/rect_7x7.json')
    ap.add_argument('--N_list',   nargs='+', type=int,
                    default=[1000, 10000, 100000, 1000000],
                    help='Shots per sigma to test (total = 3x this)')
    ap.add_argument('--K',        type=int, default=100,
                    help='Noise repetitions per (trial, N)')
    ap.add_argument('--seed',     type=int, default=42)
    ap.add_argument('--v_hop',    type=float, default=0.448)
    ap.add_argument('--thr',      type=float, default=1e-3)
    ap.add_argument('--thr_L',    type=float, default=0.05)
    ap.add_argument('--T_max',    type=int,   default=5)
    ap.add_argument('--top_k',    type=int,   default=1,
                    help='Declare top-k bonds; detection = true bond in top-k')
    ap.add_argument('--out',      default='logs/shots_needed_results.json')
    ap.add_argument('--plot',     default=None,
                    help='Path for output PNG plot (optional)')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load graph + plan once
    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))
    n_bonds = len(g['edges'])

    print(f"Graph: {args.graph}  ({len(all_sites)} sites, {n_bonds} bonds)")
    print(f"Plan:  {len(plan['placements'])} scans")
    print(f"Trials: {args.trials}")
    print(f"N_per_sigma: {args.N_list}  (total shots = 3x)")
    print(f"K={args.K} repetitions  top_k={args.top_k}\n")

    # --- Load exact log data for all trials and scans ---
    trials_data = {}
    for tid in args.trials:
        logdir = f"{args.logbase}/trial_{tid}"
        broken_path = f"{logdir}/broken.json"
        if not os.path.isfile(broken_path):
            print(f"  Trial {tid}: broken.json missing, skipping")
            continue
        br = json.load(open(broken_path))
        true_bond = br.get('bonds', [br.get('bond', -1)])[0]

        scan_data = {}
        ok = True
        for sc in plan['placements']:
            sid = sc['scan']
            prefix = sc.get('log_prefix', f'scan{sid}')
            try:
                _, _, db = parse_pauli_log(f'{logdir}/{prefix}_base.log', 'NL')
                _, _, dp = parse_pauli_log(f'{logdir}/{prefix}_pert.log', 'NL')
                scan_data[sid] = {'base': db, 'pert': dp}
            except Exception as e:
                print(f"  Trial {tid} scan {sid}: {e}, skipping trial")
                ok = False
                break
        if not ok:
            continue

        # Verify exact detection works (sanity check)
        ranked_exact = detect_arrays(scan_data, plan, g, adj, all_sites, site_idx,
                                     args.v_hop, args.thr, args.thr_L, args.T_max)
        rank_exact = rank_of_bond(ranked_exact, true_bond)
        trials_data[tid] = {'scan_data': scan_data, 'true_bond': true_bond,
                            'rank_exact': rank_exact}
        top_k_ok = rank_exact <= args.top_k
        print(f"  Trial {tid}  bond={true_bond}  exact_rank={rank_exact} "
              f"{'✓' if top_k_ok else '✗'}")

    if not trials_data:
        print("No trials loaded. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(trials_data)} trials.\n")

    # --- Shot noise study ---
    results = {}  # N -> {'detect': [...], 'fp': [...]}

    for N in args.N_list:
        print(f"N_per_sigma={N:>8d}  (total {3*N:>9d} shots) ...", flush=True)
        detect_ranks = []   # broken-bond case
        fp_ranks     = []   # false-positive case (healthy device)

        for tid, td in trials_data.items():
            scan_data = td['scan_data']
            true_bond = td['true_bond']

            for rep in range(args.K):
                # --- broken bond: inject noise into n_pert ---
                noisy_scan = {}
                for sid, entry in scan_data.items():
                    n_pert_noisy = inject_noise_otoc(entry['pert'], N, rng)
                    noisy_scan[sid] = {'base': entry['base'], 'pert': n_pert_noisy}
                ranked = detect_arrays(noisy_scan, plan, g, adj, all_sites, site_idx,
                                       args.v_hop, args.thr, args.thr_L, args.T_max)
                detect_ranks.append(rank_of_bond(ranked, true_bond))

                # --- false positive: inject noise into n_base (healthy device) ---
                fp_scan = {}
                for sid, entry in scan_data.items():
                    n_fp_noisy = inject_noise_otoc(entry['base'], N, rng)
                    fp_scan[sid] = {'base': entry['base'], 'pert': n_fp_noisy}
                ranked_fp = detect_arrays(fp_scan, plan, g, adj, all_sites, site_idx,
                                          args.v_hop, args.thr, args.thr_L, args.T_max)
                # false positive: any bond declared at rank 1
                fp_declared = ranked_fp[0][0] if ranked_fp else -1
                fp_ranks.append(fp_declared)

        detect_arr = np.array(detect_ranks)
        # Detection rate: true bond in top-k
        det_rate = np.mean(detect_arr <= args.top_k)
        # 95% CI via bootstrap
        boot = [np.mean(rng.choice(detect_arr, len(detect_arr)) <= args.top_k)
                for _ in range(1000)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        # False positive rate: any bond declared (from healthy device)
        n_fp = sum(1 for r in fp_ranks if r != -1 and r <= args.top_k)
        fp_rate = n_fp / len(fp_ranks)

        results[N] = {
            'N_per_sigma': N,
            'N_total': 3 * N,
            'detection_rate': float(det_rate),
            'ci_lo': float(ci_lo),
            'ci_hi': float(ci_hi),
            'fp_rate': float(fp_rate),
            'n_trials': len(trials_data),
            'K': args.K,
            'n_samples': len(detect_arr),
        }
        print(f"  detect={det_rate:.1%}  [{ci_lo:.1%}, {ci_hi:.1%}]  "
              f"fp={fp_rate:.1%}")

    # --- Save results ---
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'params': vars(args), 'results': results}, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # --- Summary table ---
    print(f"\n{'N_per_σ':>10}  {'N_total':>10}  {'detect':>8}  {'95% CI':>18}  {'fp':>6}")
    print("-" * 60)
    for N, r in sorted(results.items()):
        print(f"{r['N_per_sigma']:>10d}  {r['N_total']:>10d}  "
              f"{r['detection_rate']:>7.1%}  "
              f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]  "
              f"{r['fp_rate']:>5.1%}")

    # --- Plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            Ns     = sorted(results.keys())
            drates = [results[N]['detection_rate'] for N in Ns]
            cilo   = [results[N]['ci_lo'] for N in Ns]
            cihi   = [results[N]['ci_hi'] for N in Ns]
            fprats = [results[N]['fp_rate'] for N in Ns]
            Ntots  = [results[N]['N_total'] for N in Ns]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.semilogx(Ntots, drates, 'o-', color='steelblue',
                        linewidth=2, markersize=6, label='Detection rate (broken bond)')
            ax.fill_between(Ntots, cilo, cihi, alpha=0.2, color='steelblue',
                            label='95% CI')
            ax.semilogx(Ntots, fprats, 's--', color='tomato',
                        linewidth=1.5, markersize=5, label='False-positive rate')
            ax.axhline(0.9, color='gray', linestyle=':', linewidth=1)
            ax.set_xlabel('Total shots per site (3 × N per σ)')
            ax.set_ylabel('Rate')
            ax.set_title(f'Shots-needed curve — OTOC echo protocol\n'
                         f'{len(trials_data)} trials × K={args.K} reps, top-{args.top_k} detection')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='lower right')
            ax.grid(True, which='both', alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"Plot saved to {args.plot}")
        except ImportError:
            print("matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
