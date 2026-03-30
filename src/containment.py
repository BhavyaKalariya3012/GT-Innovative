"""
containment.py
--------------
Containment strategies for limiting misinformation spread on a social network.

Three complementary approaches:
  1. remove_top_k_nodes()    – deplatform the most central nodes
  2. block_edges_random()    – probabilistically drop edges (throttling)
  3. add_fact_check_nodes()  – inject immune "fact-checker" nodes that absorb spread

Public helpers:
  compare_containment()      – benchmark all strategies vs. baseline SIR
  print_containment_report() – pretty-print the comparison DataFrame
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.epidemic_model import run_sir, peak_infected


# ---------------------------------------------------------------------------
# Strategy 1 — Remove Top-K Hub Nodes
# ---------------------------------------------------------------------------

def remove_top_k_nodes(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
    k: int = 20,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Remove the *k* most influential nodes to fragment the network.

    Targets nodes with the highest *composite* centrality score — effectively
    modelling deplatforming of super-spreader accounts.

    Parameters
    ----------
    G : nx.DiGraph
        Original social network graph.
    centrality_df : pd.DataFrame
        Output of :func:`src.centrality.compute_centralities`.
    k : int
        Number of nodes to remove. Default 20.

    Returns
    -------
    G_contained : nx.DiGraph
        Copy of *G* with the top-*k* nodes removed.
    removed : list
        Node IDs that were removed.
    """
    top_nodes = centrality_df.sort_values("composite", ascending=False).head(k).index.tolist()
    removed = [n for n in top_nodes if n in G]
    G_contained = G.copy()
    G_contained.remove_nodes_from(removed)
    print(f"[containment] remove_top_k_nodes: removed {len(removed)} hub nodes.")
    return G_contained, removed


# ---------------------------------------------------------------------------
# Strategy 2 — Randomly Block Edges (throttling)
# ---------------------------------------------------------------------------

def block_edges_random(
    G: nx.DiGraph,
    probability: float = 0.30,
    seed: int = 42,
) -> Tuple[nx.DiGraph, int]:
    """Drop each edge with *probability* to simulate platform throttling.

    Models content delivery restrictions — not all edges carry information
    at full speed (e.g., reduced algorithmic amplification).

    Parameters
    ----------
    G : nx.DiGraph
        Original social network graph.
    probability : float
        Probability of removing each edge. Range (0, 1). Default 0.30.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    G_contained : nx.DiGraph
        Pruned copy of *G*.
    n_removed : int
        Number of edges removed.
    """
    rng = np.random.default_rng(seed)
    G_contained = G.copy()
    edges_to_remove = [
        (u, v) for u, v in G.edges()
        if rng.random() < probability
    ]
    G_contained.remove_edges_from(edges_to_remove)
    n_removed = len(edges_to_remove)
    print(f"[containment] block_edges_random: removed {n_removed:,} edges "
          f"({probability*100:.0f}% throttle).")
    return G_contained, n_removed


# ---------------------------------------------------------------------------
# Strategy 3 — Add Fact-Check Nodes
# ---------------------------------------------------------------------------

def add_fact_check_nodes(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
    k: int = 20,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Inject *k* fact-checker nodes that absorb spread from hub neighbours.

    Fact-checker nodes are connected to the top-*k* hub nodes and are
    initialised as permanently Recovered (immune) in downstream SIR runs,
    effectively forming a firewall around the highest-risk spreaders.

    Implementation note
    -------------------
    The returned graph has new string-keyed nodes (``'fc_0'``, ``'fc_1'``, …).
    When passing *seeds* to :func:`run_sir`, ensure these nodes are excluded
    from the infected seed set (they are naturally immune).

    Parameters
    ----------
    G : nx.DiGraph
        Original social network graph.
    centrality_df : pd.DataFrame
        Output of :func:`src.centrality.compute_centralities`.
    k : int
        Number of fact-check nodes to inject. Default 20.

    Returns
    -------
    G_contained : nx.DiGraph
        Augmented copy of *G* with fact-check nodes added.
    fc_nodes : list
        IDs of the injected fact-check nodes.
    """
    G_contained = G.copy()
    top_hubs = centrality_df.sort_values("composite", ascending=False).head(k).index.tolist()
    fc_nodes = []

    for i, hub in enumerate(top_hubs):
        if hub not in G_contained:
            continue
        fc_id = f"fc_{i}"
        G_contained.add_node(fc_id, fact_checker=True)
        # Bidirectional link so fact-checker can receive and block spread
        G_contained.add_edge(hub, fc_id)
        G_contained.add_edge(fc_id, hub)
        fc_nodes.append(fc_id)

    print(f"[containment] add_fact_check_nodes: injected {len(fc_nodes)} "
          f"fact-checker nodes linked to top hubs.")
    return G_contained, fc_nodes


# ---------------------------------------------------------------------------
# Benchmarking — Compare all strategies vs. baseline
# ---------------------------------------------------------------------------

def compare_containment(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
    seeds: List[Any],
    beta: float = 0.30,
    gamma: float = 0.10,
    k: int = 20,
    edge_block_prob: float = 0.30,
    steps: int = 100,
    runs: int = 5,
) -> pd.DataFrame:
    """Run all three containment strategies + baseline and compare results.

    Each strategy is evaluated over *runs* independent SIR Monte-Carlo runs.
    Returns a summary DataFrame with mean/std peak infected and % spread reduction.

    Parameters
    ----------
    G : nx.DiGraph
        Original social network.
    centrality_df : pd.DataFrame
        Pre-computed centrality table.
    seeds : list
        Initial infected node IDs.
    beta : float
        SIR infection probability.
    gamma : float
        SIR recovery probability.
    k : int
        Budget for node-removal and fact-check strategies.
    edge_block_prob : float
        Edge removal probability for throttling strategy.
    steps : int
        Max simulation steps.
    runs : int
        Monte-Carlo repetitions per strategy.

    Returns
    -------
    pd.DataFrame
        Indexed by strategy name. Columns:
        ``mean_peak``, ``std_peak``, ``reduction_pct``.
    """

    def _avg_peak(graph: nx.DiGraph, seed_nodes: List[Any]) -> Tuple[float, float]:
        """Run *runs* SIR simulations and return (mean, std) peak infected."""
        effective = [s for s in seed_nodes if s in graph]
        if not effective:
            # Fall back: pick highest in-degree node
            if graph.number_of_nodes() > 0:
                effective = [max(dict(graph.in_degree()).items(), key=lambda x: x[1])[0]]
            else:
                return 0.0, 0.0
        peaks = []
        for _ in range(runs):
            hist, _ = run_sir(graph, effective, beta=beta, gamma=gamma, steps=steps)
            peaks.append(peak_infected(hist))
        return float(np.mean(peaks)), float(np.std(peaks))

    results: Dict[str, Dict[str, float]] = {}

    # ── Baseline ──────────────────────────────────────────────────────────
    print("[containment] Running baseline …")
    b_mean, b_std = _avg_peak(G, seeds)
    results["baseline"] = {"mean_peak": b_mean, "std_peak": b_std, "reduction_pct": 0.0}

    # ── Strategy 1: Remove top-K nodes ────────────────────────────────────
    print("[containment] Running remove_top_k_nodes …")
    G1, _ = remove_top_k_nodes(G, centrality_df, k=k)
    m1, s1 = _avg_peak(G1, seeds)
    results["remove_hubs"] = {
        "mean_peak": m1,
        "std_peak":  s1,
        "reduction_pct": max(0.0, (b_mean - m1) / max(b_mean, 1) * 100),
    }

    # ── Strategy 2: Block edges randomly ─────────────────────────────────
    print("[containment] Running block_edges_random …")
    G2, _ = block_edges_random(G, probability=edge_block_prob)
    m2, s2 = _avg_peak(G2, seeds)
    results["edge_block"] = {
        "mean_peak": m2,
        "std_peak":  s2,
        "reduction_pct": max(0.0, (b_mean - m2) / max(b_mean, 1) * 100),
    }

    # ── Strategy 3: Fact-check nodes ─────────────────────────────────────
    print("[containment] Running add_fact_check_nodes …")
    G3, fc_nodes = add_fact_check_nodes(G, centrality_df, k=k)
    # fact-check nodes start as recovered (immune) — exclude from seeds
    fc_set = set(fc_nodes)
    seeds3 = [s for s in seeds if s not in fc_set]
    m3, s3 = _avg_peak(G3, seeds3)
    results["fact_check"] = {
        "mean_peak": m3,
        "std_peak":  s3,
        "reduction_pct": max(0.0, (b_mean - m3) / max(b_mean, 1) * 100),
    }

    df = pd.DataFrame(results).T
    df.index.name = "strategy"
    return df


def print_containment_report(df: pd.DataFrame) -> None:
    """Pretty-print the containment comparison DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compare_containment`.
    """
    print("\n" + "=" * 60)
    print("  🛡️  Containment Strategy Comparison")
    print("=" * 60)
    print(f"  {'Strategy':<18} {'Mean Peak':>10} {'Std':>8} {'Reduction %':>13}")
    print("-" * 60)
    for strategy, row in df.iterrows():
        tag = " ← best" if row["reduction_pct"] == df["reduction_pct"].max() and strategy != "baseline" else ""
        print(f"  {strategy:<18} {row['mean_peak']:>10.1f} {row['std_peak']:>8.1f} "
              f"{row['reduction_pct']:>11.1f}%{tag}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.graph_builder import load_graph
    from src.centrality import compute_centralities

    G = load_graph()
    cdf = compute_centralities(G, k=min(200, G.number_of_nodes()))
    seeds = list(G.nodes())[:3]

    results = compare_containment(G, cdf, seeds, runs=3)
    print_containment_report(results)
