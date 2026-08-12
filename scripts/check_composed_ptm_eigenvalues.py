#!/usr/bin/env python3
"""
check_composed_ptm_eigenvalues.py
Read precomputed noisy PTMs from HPP files and analyse their eigenvalues.
These are the FULL composed PTMs: G1·Mn·G2·Mn·G3·Mn·G4 per bond.

Usage:
  python3 scripts/check_composed_ptm_eigenvalues.py
"""

import re
import numpy as np

def parse_hpp_array(path, array_name):
    """
    Extract float array data[n_bonds][256] from a C++ HPP file.
    Returns numpy array shape (n_bonds, 16, 16) in column-major order.
    Uses brace-counting to extract exactly the right content.
    """
    with open(path) as f:
        text = f.read()

    # Find declaration line to get n_bonds
    decl = re.search(
        rf'static const float {re.escape(array_name)}\[(\d+)\]\[256\]\s*=\s*\{{',
        text)
    if not decl:
        raise ValueError(f"Array {array_name} not found in {path}")

    n_bonds = int(decl.group(1))

    # Walk forward from the opening brace, counting depth
    start = decl.end() - 1          # position of the opening '{'
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    raw = text[start+1:i]            # content between outer braces

    # Parse each bond's inner {…} separately — avoids comment/whitespace ambiguity
    bond_blocks = re.findall(r'\{([^{}]+)\}', raw)
    if len(bond_blocks) != n_bonds:
        raise AssertionError(f"Expected {n_bonds} bond blocks, got {len(bond_blocks)}")
    float_re = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?f')
    vals = []
    for blk in bond_blocks:
        vals.extend(float(x.rstrip('f')) for x in float_re.findall(blk))
    if len(vals) != n_bonds * 256:
        raise AssertionError(f"Expected {n_bonds*256} values, got {len(vals)}")

    arr = np.array(vals, dtype=np.float64).reshape(n_bonds, 256)
    # Each 256-element row is a 16x16 PTM stored column-major
    ptms = np.array([arr[b].reshape(16, 16, order='F') for b in range(n_bonds)])
    return ptms  # shape (n_bonds, 16, 16)


def summarise(label, ptms):
    """Print eigenvalue statistics across all bonds."""
    n_bonds = ptms.shape[0]
    all_eigs = []
    neg_count  = 0
    large_count = 0
    max_offdiag = []
    max_imag    = []

    for b in range(n_bonds):
        M = ptms[b]
        eigs = np.linalg.eigvals(M)
        all_eigs.extend(eigs.tolist())
        neg_count   += (np.real(eigs) < 0).sum()
        large_count += (np.abs(eigs) > 1.01).sum()
        offdiag = M - np.diag(np.diag(M))
        max_offdiag.append(np.max(np.abs(offdiag)))
        max_imag.append(np.max(np.abs(np.imag(eigs))))

    all_eigs = np.array(all_eigs)
    er = np.real(all_eigs)
    ei = np.imag(all_eigs)

    print(f"\n{'='*60}")
    print(f"{label}  ({n_bonds} bonds, {n_bonds*16} eigenvalues total)")
    print(f"  Re(eig):     min={er.min():.4f}  max={er.max():.4f}  "
          f"mean={er.mean():.4f}")
    print(f"  |Im(eig)|:   max={np.max(np.abs(ei)):.4f}  "
          f"mean={np.mean(np.abs(ei)):.6f}")
    print(f"  neg Re(eig): {neg_count} / {len(all_eigs)}  "
          f"({100*neg_count/len(all_eigs):.1f}%)")
    print(f"  |eig|>1.01:  {large_count} / {len(all_eigs)}")
    print(f"  max off-diag per bond: "
          f"min={min(max_offdiag):.4f}  max={max(max_offdiag):.4f}  "
          f"mean={np.mean(max_offdiag):.4f}")

    # Per-bond diagonal vs off-diagonal content
    diag_power    = np.mean([np.sum(np.diag(ptms[b])**2) for b in range(n_bonds)])
    offdiag_power = np.mean([np.sum((ptms[b]-np.diag(np.diag(ptms[b])))**2)
                             for b in range(n_bonds)])
    print(f"  Mean diag power:    {diag_power:.4f}")
    print(f"  Mean off-diag power:{offdiag_power:.4f}  "
          f"(fraction={offdiag_power/(diag_power+offdiag_power):.3f})")

    return er, ei, max_offdiag


# ── Heavy-hex ─────────────────────────────────────────────────────────────────
print("\n=== HEAVY-HEX 3x3 ===")

ideal_ptms = parse_hpp_array(
    'include/heavyhex_ideal_kick_eps010_data.hpp',
    'HEAVY_HEX_3x3_IDEAL_KICK_EPS010')
summarise('Ideal (noiseless)', ideal_ptms)

for gamma in [1, 2, 3]:
    arr_name = f'HEAVY_HEX_3x3_NOISY_EPS010_GAMMA{gamma}_GAMMA{gamma}_0'
    path     = f'include/heavyhex_noisy_eps010_gamma{gamma}_data.hpp'
    try:
        ptms = parse_hpp_array(path, arr_name)
        summarise(f'Noisy gamma={gamma}', ptms)
    except ValueError as e:
        # Try alternate name
        arr_name2 = f'HEAVY_HEX_3x3_NOISY_EPS010_GAMMA{gamma}'
        try:
            ptms = parse_hpp_array(path, arr_name2)
            summarise(f'Noisy gamma={gamma} (alt name)', ptms)
        except ValueError:
            print(f"  Could not find array for gamma={gamma}: {e}")
            # List available array names
            with open(path) as f:
                names = re.findall(r'static const float (\w+)\[', f.read())
            print(f"  Available arrays: {names[:5]}")

# ── Rect 7x7 ──────────────────────────────────────────────────────────────────
print("\n=== RECT 7x7 ===")

ideal_ptms_r = parse_hpp_array(
    'include/rect_7x7_ideal_kick_eps010_data.hpp',
    'RECT_7x7_IDEAL_KICK_EPS010')
summarise('Ideal (noiseless)', ideal_ptms_r)

for gamma in [1, 2, 3]:
    path = f'include/rect_7x7_noisy_eps010_gamma{gamma}_data.hpp'
    with open(path) as f:
        names = re.findall(r'static const float (\w+)\[\d+\]\[256\]', f.read())
    if not names:
        print(f"  No [n][256] array found in {path}")
        continue
    arr_name = names[0]
    print(f"  Using array: {arr_name}")
    ptms = parse_hpp_array(path, arr_name)
    summarise(f'Noisy gamma={gamma}', ptms)
