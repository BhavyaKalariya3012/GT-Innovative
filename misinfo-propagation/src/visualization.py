"""
visualization.py
----------------
Network visualization utilities for the Misinformation Propagation Simulator.

Provides two rendering backends:
  1. Matplotlib (static PNG) — always available, saves file to disk
  2. PyVis  (interactive HTML) — mirroring the Streamlit app's approach,
     extended with influencer (yellow) highlighting

Public functions:
  draw_network_matplotlib()   – spring-layout PNG coloured by node state
  build_pyvis_advanced()      – PyVis HTML with state + influencer colours
  save_matplotlib_png()       – convenience wrapper that saves PNG to output/
"""

from __future__ import annotations

import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Set

import networkx as nx

# ── Matplotlib import with graceful fallback ──────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (works headlessly)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False

# ── PyVis import with graceful fallback ───────────────────────────────────
try:
    from pyvis.network import Network as PyvisNetwork
    _PYVIS_OK = True
except ImportError:
    _PYVIS_OK = False


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLOURS = {
    "S":   "#64748b",   # gray  — Susceptible
    "I":   "#ef4444",   # red   — Infected
    "R":   "#22c55e",   # green — Recovered / Immune
    "IMM": "#22c55e",   # green — Immunized (alias)
    "INF": "#facc15",   # yellow — Influencer
    "FC":  "#a78bfa",   # purple — Fact-checker node
}

_LEGEND_LABELS = {
    "S":   "Susceptible",
    "I":   "Infected",
    "R":   "Recovered / Immune",
    "INF": "Influencer (Top-K)",
    "FC":  "Fact-Checker",
}


def _node_colour(
    node: Any,
    state: Dict[Any, str],
    influencer_set: Set[Any],
    fc_set: Set[Any],
) -> str:
    """Resolve display colour for a single node."""
    if str(node).startswith("fc_") or node in fc_set:
        return _COLOURS["FC"]
    if node in influencer_set:
        return _COLOURS["INF"]
    s = state.get(node, "S")
    return _COLOURS.get(s, _COLOURS["S"])


# ---------------------------------------------------------------------------
# Backend 1 — Matplotlib static PNG
# ---------------------------------------------------------------------------

def draw_network_matplotlib(
    G: nx.DiGraph,
    state: Dict[Any, str],
    influencers: Optional[List[Any]] = None,
    fc_nodes: Optional[List[Any]] = None,
    title: str = "Misinformation Propagation Network",
    max_nodes: int = 400,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Draw and optionally save a static network visualization.

    Node colours:
    - 🔴 Red    : Infected (I)
    - 🟢 Green  : Recovered / Immune (R / IMM)
    - 🟡 Yellow : Top-K Influencer
    - 🟣 Purple : Fact-Checker node
    - ⚫ Gray   : Susceptible (S)

    Parameters
    ----------
    G : nx.DiGraph
        Social network graph.
    state : dict
        Node → state ('S', 'I', 'R') from epidemic simulation.
    influencers : list, optional
        Node IDs to highlight as influencers (yellow).
    fc_nodes : list, optional
        Node IDs that are fact-checker nodes (purple).
    title : str
        Figure title.
    max_nodes : int
        Max nodes to render (random sample for large graphs). Default 400.
    seed : int
        Layout seed for reproducibility.
    output_path : str, optional
        If provided, save the figure to this path as PNG.

    Returns
    -------
    str or None
        Absolute path to the saved PNG, or None if matplotlib unavailable.
    """
    if not _MATPLOTLIB_OK:
        print("[visualization] matplotlib not available — skipping PNG render.")
        return None

    influencer_set: Set[Any] = set(influencers or [])
    fc_set: Set[Any]         = set(fc_nodes      or [])

    # Sample for large graphs
    all_nodes = list(G.nodes())
    if len(all_nodes) > max_nodes:
        sampled = set(random.Random(seed).sample(all_nodes, max_nodes))
        # Always include influencers in the sample
        sampled |= influencer_set & set(all_nodes)
        subG = G.subgraph(sampled).copy()
    else:
        subG = G

    node_list   = list(subG.nodes())
    node_colors = [_node_colour(n, state, influencer_set, fc_set) for n in node_list]
    node_sizes  = [
        180 if n in influencer_set else
        140 if (str(n).startswith("fc_") or n in fc_set) else
        40
        for n in node_list
    ]

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    print(f"[visualization] Computing spring layout for {subG.number_of_nodes()} nodes …")
    pos = nx.spring_layout(subG, seed=seed, k=0.6)

    # Draw edges
    nx.draw_networkx_edges(
        subG, pos,
        ax=ax,
        alpha=0.15,
        edge_color="#334155",
        arrows=False,
        width=0.4,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        subG, pos,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
        alpha=0.88,
        linewidths=0.5,
        edgecolors="#1e293b",
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=_COLOURS["I"],   label="Infected"),
        mpatches.Patch(color=_COLOURS["R"],   label="Recovered / Immune"),
        mpatches.Patch(color=_COLOURS["S"],   label="Susceptible"),
        mpatches.Patch(color=_COLOURS["INF"], label="Top Influencer"),
        mpatches.Patch(color=_COLOURS["FC"],  label="Fact-Checker"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
        fontsize=9,
        framealpha=0.9,
    )

    ax.set_title(title, color="#f1f5f9", fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[visualization] PNG saved → {output_path}")
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None


def save_matplotlib_png(
    G: nx.DiGraph,
    state: Dict[Any, str],
    influencers: Optional[List[Any]] = None,
    fc_nodes: Optional[List[Any]] = None,
    filename: str = "network_state.png",
) -> Optional[str]:
    """Convenience wrapper: save a matplotlib network PNG to ``output/<filename>``.

    Parameters
    ----------
    G : nx.DiGraph
        Social network.
    state : dict
        Node → state mapping.
    influencers : list, optional
        Influencer node IDs (yellow).
    fc_nodes : list, optional
        Fact-checker node IDs (purple).
    filename : str
        Output filename (placed in ./output/ directory).

    Returns
    -------
    str or None
        Absolute path to saved PNG.
    """
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir  = os.path.join(base_dir, "output")
    output_path = os.path.join(output_dir, filename)

    return draw_network_matplotlib(
        G, state,
        influencers=influencers,
        fc_nodes=fc_nodes,
        title="Misinformation Propagation — Final Network State",
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Backend 2 — PyVis interactive HTML
# ---------------------------------------------------------------------------

def build_pyvis_advanced(
    G: nx.DiGraph,
    state: Dict[Any, str],
    influencers: Optional[List[Any]] = None,
    fc_nodes: Optional[List[Any]] = None,
    max_nodes: int = 300,
    seed: int = 42,
) -> Optional[str]:
    """Build an interactive PyVis HTML network with full colour coding.

    Node colours:
    - 🔴 Red    : Infected
    - 🟢 Green  : Recovered / Immune
    - 🟡 Yellow : Top-K Influencer
    - 🟣 Purple : Fact-Checker
    - ⚫ Gray   : Susceptible

    Parameters
    ----------
    G : nx.DiGraph
        Social network graph.
    state : dict
        Node → state mapping.
    influencers : list, optional
        Influencer node IDs.
    fc_nodes : list, optional
        Fact-checker node IDs.
    max_nodes : int
        Max nodes to render. Default 300.
    seed : int
        Random seed for sampling.

    Returns
    -------
    str or None
        HTML string, or None if PyVis is unavailable.
    """
    if not _PYVIS_OK:
        print("[visualization] pyvis not available — skipping HTML render.")
        return None

    influencer_set: Set[Any] = set(influencers or [])
    fc_set: Set[Any]         = set(fc_nodes      or [])

    all_nodes = list(G.nodes())
    if len(all_nodes) > max_nodes:
        sampled = set(random.Random(seed).sample(all_nodes, max_nodes))
        sampled |= influencer_set & set(all_nodes)
        subG = G.subgraph(sampled).copy()
    else:
        subG = G

    net = PyvisNetwork(
        height="560px",
        width="100%",
        bgcolor="#0d1117",
        font_color="#e2e8f0",
        directed=True,
    )
    net.repulsion(node_distance=85, spring_length=110)

    for node in subG.nodes():
        colour = _node_colour(node, state, influencer_set, fc_set)
        s      = state.get(node, "S")

        if str(node).startswith("fc_") or node in fc_set:
            label = f"FC-{node}"
            size  = 16
        elif node in influencer_set:
            label = f"★ {node}"
            size  = 14
        else:
            label = str(node)
            size  = 7

        title = (
            f"Node: {node}\n"
            f"State: {s}\n"
            f"{'[Influencer]' if node in influencer_set else ''}"
            f"{'[Fact-Checker]' if (str(node).startswith('fc_') or node in fc_set) else ''}"
        )

        net.add_node(
            str(node),
            label=label,
            color=colour,
            title=title,
            size=size,
            borderWidth=1,
            borderWidthSelected=3,
        )

    for u, v in subG.edges():
        if str(u) in net.get_nodes() and str(v) in net.get_nodes():
            net.add_edge(str(u), str(v), color="#334155", width=0.5, arrows="to")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        tmp_path = f.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    os.unlink(tmp_path)
    return html_content


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.graph_builder import load_graph
    from src.centrality import compute_centralities
    from src.epidemic_model import run_sir

    G   = load_graph()
    cdf = compute_centralities(G, k=min(200, G.number_of_nodes()))
    top = cdf.head(10).index.tolist()
    seeds = list(G.nodes())[:3]
    _, state = run_sir(G, seeds, beta=0.3, gamma=0.1, steps=50)

    save_matplotlib_png(G, state, influencers=top, filename="demo_network.png")
    print("Done. Check output/demo_network.png")
