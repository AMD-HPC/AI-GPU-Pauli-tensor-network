"""
plot_lightcone.py — Visualize operator lightcone on heavy_hex_3x3

Reads spatial Pauli weight data from main_heavyhex --spatial=file.dat
and produces:
  - Multi-panel figure: one panel per cycle showing weight spreading
  - Optional GIF animation
  - Honeycomb-aware layout: sites colored by weight, hexagon interiors = zero

Usage:
  # Single run — multi-panel
  python3 plot_lightcone.py --data logs/chi_scaling/spatial_chi200.dat

  # Compare multiple runs (different chi or center sites)
  python3 plot_lightcone.py --data logs/chi_scaling/spatial_chi200.dat \
                                    logs/chi_scaling/spatial_chi300.dat \
                                    logs/chi_scaling/spatial_chi400.dat \
                             --labels "χ=200" "χ=300" "χ=400"

  # GIF animation
  python3 plot_lightcone.py --data logs/lightcone_sites/spatial_site2.dat --gif

  # Custom graph JSON (for other graphs)
  python3 plot_lightcone.py --data spatial.dat --graph data/graphs/heavy_hex_3x3.json
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from collections import defaultdict
import os

# ── Default graph path ────────────────────────────────────────────────────────
DEFAULT_GRAPH = os.path.join(
    os.path.dirname(__file__),
    "../../QC/amd-TensorNetworkQuantumSimulator.jl/data/graphs/heavy_hex_3x3.json"
)
if not os.path.exists(DEFAULT_GRAPH):
    DEFAULT_GRAPH = "data/graphs/heavy_hex_3x3.json"

# ── Load graph and compute layout ─────────────────────────────────────────────

def load_graph(graph_path):
    with open(graph_path) as f:
        d = json.load(f)
    G = nx.Graph()
    vertices = [v[0] for v in d["vertices"]]
    G.add_nodes_from(vertices)
    for e in d["edges"]:
        u, v = e["edge"][0][0], e["edge"][1][0]
        G.add_edge(u, v)
    degree = dict(G.degree())
    return G, vertices, degree

def compute_layout(G, degree):
    """Kamada-Kawai with boundary degree-2 nodes fixed to hex positions."""
    pos = nx.kamada_kawai_layout(G, scale=5.0)

    # Fix boundary degree-2 nodes (single degree-3 neighbor)
    interior_lengths = []
    for n in G.nodes():
        if degree[n] == 2:
            nbrs = list(G.neighbors(n))
            deg3_nbrs = [nb for nb in nbrs if degree[nb] == 3]
            if len(deg3_nbrs) == 2:
                p0, p1, p2 = pos[n], pos[deg3_nbrs[0]], pos[deg3_nbrs[1]]
                interior_lengths.append((
                    np.hypot(p0[0]-p1[0], p0[1]-p1[1]) +
                    np.hypot(p0[0]-p2[0], p0[1]-p2[1])
                ) / 2)
    edge_len = np.mean(interior_lengths) if interior_lengths else 1.0

    for n in G.nodes():
        if degree[n] != 2: continue
        nbrs = list(G.neighbors(n))
        deg3_nbrs = [nb for nb in nbrs if degree[nb] == 3]
        if len(deg3_nbrs) != 1: continue
        anchor = deg3_nbrs[0]
        ax, ay = pos[anchor]
        other = [nb for nb in G.neighbors(anchor) if nb != n]
        if not other: continue
        cx = np.mean([pos[nb][0] for nb in other])
        cy = np.mean([pos[nb][1] for nb in other])
        dx, dy = ax - cx, ay - cy
        norm = np.hypot(dx, dy) or 1.0
        pos[n] = (ax + dx/norm * edge_len, ay + dy/norm * edge_len)

    return pos

# ── Parse spatial data file ───────────────────────────────────────────────────

def parse_spatial(path):
    """
    Returns list of (cycle, C, chi, weights_array) tuples.
    weights_array[site_id] = Pauli weight.
    """
    cycles = []
    current = None
    n_sites = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("# heavy_hex"):
                # Parse n_sites from header
                for part in line.split():
                    if part.startswith("n_sites="):
                        n_sites = int(part.split("=")[1])
            elif line.startswith("# cycle"):
                # "# cycle 1  C=0.886203  chi=4"
                parts = line.split()
                cyc = int(parts[2])
                C   = float(parts[3].split("=")[1])
                chi = int(parts[4].split("=")[1])
                if n_sites is None: n_sites = 68  # fallback
                current = {"cycle": cyc, "C": C, "chi": chi,
                           "weights": np.zeros(n_sites)}
                cycles.append(current)
            elif line.startswith("#"):
                continue
            elif current is not None:
                sid, w = int(line.split()[0]), float(line.split()[1])
                current["weights"][sid] = w

    return cycles

# ── Single panel drawing ───────────────────────────────────────────────────────

def draw_panel(ax, G, pos, degree, weights, center, title,
               cmap="hot", vmin=0, vmax=1, show_labels=False):
    """Draw one cycle's Pauli weight on the graph."""
    # Edge colors: gray
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc",
                           width=1.5, alpha=0.7)

    # Node colors by weight
    node_list = list(G.nodes())
    # Graph uses 1-based vertex ids, weights array is 0-based (CppSim site ids)
    node_colors = [weights[n-1] for n in node_list]
    node_sizes  = [300 if degree[n] == 3 else 150 for n in node_list]

    sc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                nodelist=node_list,
                                node_color=node_colors,
                                node_size=node_sizes,
                                cmap=cmap,
                                vmin=vmin, vmax=vmax,
                                edgecolors="#444444",
                                linewidths=0.8)

    # Mark center site (center is 0-based, graph nodes are 1-based)
    center_node = center + 1  # CppSim 0-based → graph 1-based
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           nodelist=[center_node],
                           node_color=[weights[center]],
                           node_size=[400 if degree[center_node]==3 else 200],
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           edgecolors="cyan", linewidths=2.5)

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=5,
                                font_color="#222222")

    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")
    return sc

# ── Multi-panel figure ────────────────────────────────────────────────────────

def plot_multipanel(cycles, G, pos, degree, center, out_path,
                    label="", cmap="hot", n_cols=5):
    n = len(cycles)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 3.0))
    axes = np.array(axes).flatten()

    # Use per-panel normalization so spreading is visible at all cycles
    sc = None
    for i, cyc in enumerate(cycles):
        vmax_i = max(max(cyc["weights"]), 1e-6)
        title = (f"Cycle {cyc['cycle']}  χ={cyc['chi']}\n"
                 f"C(t)={cyc['C']:.3f}")
        sc = draw_panel(axes[i], G, pos, degree,
                        cyc["weights"], center, title,
                        cmap=cmap, vmin=0, vmax=vmax_i)
    vmax = max(max(c["weights"]) for c in cycles)
    vmax = max(vmax, 1e-6)

    # Hide unused panels
    for i in range(len(cycles), len(axes)):
        axes[i].axis("off")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Non-identity Pauli weight n(v,t)")

    sup = f"Operator lightcone — heavy_hex_3x3  center=site{center}"
    if label: sup += f"  {label}"
    fig.suptitle(sup, fontsize=11, y=1.01)

    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ── GIF animation ─────────────────────────────────────────────────────────────

def plot_gif(cycles, G, pos, degree, center, out_path, cmap="hot"):
    try:
        import imageio
    except ImportError:
        print("imageio not installed — skipping GIF. pip install imageio")
        return

    frames = []

    for cyc in cycles:
        vmax = max(max(cyc["weights"]), 1e-6)  # per-frame normalization
        fig, ax = plt.subplots(figsize=(7, 6))
        title = (f"Cycle {cyc['cycle']}  χ={cyc['chi']}  C(t)={cyc['C']:.4f}")
        draw_panel(ax, G, pos, degree, cyc["weights"], center, title,
                   cmap=cmap, vmin=0, vmax=vmax)

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=mcolors.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="n(v,t)", shrink=0.8)
        fig.suptitle(f"heavy_hex_3x3  center=site{center}", fontsize=10)

        import io
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)  # no bbox_inches="tight" → fixed size
        plt.close()
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))

    imageio.v2.mimsave(out_path, frames, fps=2, loop=0)
    print(f"Saved GIF: {out_path}")

# ── Compare multiple runs side by side ────────────────────────────────────────

def plot_compare(all_cycles, labels, G, pos, degree, center, out_path,
                 cmap="hot", cycle_indices=None):
    """One row per dataset, one column per selected cycle."""
    if cycle_indices is None:
        # Pick 5 evenly spaced cycles
        n = min(len(c) for c in all_cycles)
        cycle_indices = [int(i) for i in np.linspace(0, n-1, 5)]

    n_rows = len(all_cycles)
    n_cols = len(cycle_indices)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.0, n_rows * 3.2))
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    # Per-row normalization so each dataset is readable independently
    for row, (cycs, label) in enumerate(zip(all_cycles, labels)):
        vmax_row = max(max(cycs[ci]["weights"]) for ci in cycle_indices if ci < len(cycs))
        vmax_row = max(vmax_row, 1e-6)
        for col, ci in enumerate(cycle_indices):
            if ci >= len(cycs): continue
            cyc = cycs[ci]
            title = f"Cycle {cyc['cycle']}  χ={cyc['chi']}"
            if col == 0:
                title = f"{label}\n{title}"
            draw_panel(axes[row][col], G, pos, degree,
                       cyc["weights"], center, title,
                       cmap=cmap, vmin=0, vmax=vmax_row)
    vmax = max(max(max(c["weights"]) for c in cycs) for cycs in all_cycles)
    vmax = max(vmax, 1e-6)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="n(v,t)")

    fig.suptitle(f"Operator lightcone comparison — center=site{center}",
                 fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize operator lightcone on heavy_hex lattice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--data", nargs="+", required=True,
                        help="Spatial data file(s) from main_heavyhex --spatial=")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each data file (for comparison plot)")
    parser.add_argument("--graph", default=DEFAULT_GRAPH,
                        help="Path to heavy_hex JSON graph")
    parser.add_argument("--center", type=int, default=None,
                        help="Center site id (auto-detected from data file if not set)")
    parser.add_argument("--gif", action="store_true",
                        help="Also produce GIF animation for first data file")
    parser.add_argument("--cmap", default="hot",
                        help="Colormap (default: hot)")
    parser.add_argument("--cycles", nargs="+", type=int, default=None,
                        help="Which cycles to show in comparison plot (0-based index)")
    parser.add_argument("--out", default=None,
                        help="Output filename prefix (default: lightcone)")
    args = parser.parse_args()

    # Load graph
    G, vertices, degree = load_graph(args.graph)
    pos = compute_layout(G, degree)

    # Load data
    all_cycles = [parse_spatial(f) for f in args.data]
    labels = args.labels or [os.path.basename(f).replace(".dat","")
                              for f in args.data]

    # Detect center from first file
    center = args.center
    if center is None:
        # Find site with weight=1 at cycle 0
        if all_cycles and all_cycles[0]:
            w = all_cycles[0][0]["weights"]
            center = int(np.argmax(w))
        else:
            center = 2

    out_prefix = args.out or "lightcone"

    if len(args.data) == 1:
        # Single file: multi-panel
        out_panel = f"{out_prefix}_panel.png"
        plot_multipanel(all_cycles[0], G, pos, degree, center,
                        out_panel, label=labels[0], cmap=args.cmap)
        if args.gif:
            out_gif = f"{out_prefix}.gif"
            plot_gif(all_cycles[0], G, pos, degree, center,
                     out_gif, cmap=args.cmap)
    else:
        # Multiple files: comparison
        out_compare = f"{out_prefix}_compare.png"
        plot_compare(all_cycles, labels, G, pos, degree, center,
                     out_compare, cmap=args.cmap, cycle_indices=args.cycles)
        # Also individual panels
        for cycs, label, path in zip(all_cycles, labels, args.data):
            stem = os.path.basename(path).replace(".dat","")
            plot_multipanel(cycs, G, pos, degree, center,
                            f"{out_prefix}_{stem}.png",
                            label=label, cmap=args.cmap)
        if args.gif:
            for cycs, label, path in zip(all_cycles, labels, args.data):
                stem = os.path.basename(path).replace(".dat","")
                plot_gif(cycs, G, pos, degree, center,
                         f"{out_prefix}_{stem}.gif", cmap=args.cmap)

if __name__ == "__main__":
    main()
