"""
immunization.py
---------------
Six immunization strategies to reduce misinformation spread and a
benchmarking utility that compares them via repeated SIR simulations.
"""

import random
from typing import List, Tuple, Any, Dict

import networkx as nx
import pandas as pd
import numpy as np

from src.epidemic_model import run_sir, peak_infected


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _remove_nodes(G: nx.DiGraph, targets: List[Any]) -> nx.DiGraph:
    """Return a copy of *G* with *targets* removed.

    Parameters
    ----------
    G : nx.DiGraph
        Original graph.
    targets : list
        Nodes to remove (immunize).

    Returns
    -------
    nx.DiGraph
        Copy of the graph with the targeted nodes removed.
    """
    G_copy = G.copy()
    G_copy.remove_nodes_from([n for n in targets if n in G_copy])
    return G_copy


def _safe_nodes(G: nx.DiGraph, candidates: List[Any], budget: int) -> List[Any]:
    """Return up to *budget* nodes from *candidates* that exist in *G*.

    Parameters
    ----------
    G : nx.DiGraph
        Reference graph.
    candidates : list
        Candidate node list (pre-sorted by priority).
    budget : int
        Maximum number of nodes to return.

    Returns
    -------
    list
    """
    existing = [n for n in candidates if n in G]
    return existing[:budget]


# ---------------------------------------------------------------------------
# Six immunization strategies
# ---------------------------------------------------------------------------

def random_immunization(
    G: nx.DiGraph,
    budget: int,
    centrality_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Select *budget* random nodes for immunization.

    Parameters
    ----------
    G : nx.DiGraph
        Contact network.
    budget : int
        Number of nodes to immunize.
    centrality_df : pd.DataFrame
        Pre-computed centrality table (unused but kept for API consistency).

    Returns
    -------
    G_immunized : nx.DiGraph
        Graph with targeted nodes removed.
    targets : list
        Immunized node IDs.
    """
    nodes = list(G.nodes())
    targets = random.sample(nodes, min(budget, len(nodes)))
    return _remove_nodes(G, targets), targets


def degree_immunization(
    G: nx.DiGraph,
    budget: int,
    centrality_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Immunize nodes with the highest *degree* centrality.

    Parameters
    ----------
    G : nx.DiGraph
        Contact network.
    budget : int
        Number of nodes to immunize.
    centrality_df : pd.DataFrame
        Pre-computed centrality table with a ``degree`` column.

    Returns
    -------
    G_immunized : nx.DiGraph
    targets : list
    """
    sorted_nodes = centrality_df.sort_values("degree", ascending=False).index.tolist()
    targets = _safe_nodes(G, sorted_nodes, budget)
    return _remove_nodes(G, targets), targets


def betweenness_immunization(
    G: nx.DiGraph,
    budget: int,
    centrality_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Immunize nodes with the highest *betweenness* centrality.

    Parameters
    ----------
    G : nx.DiGraph
        Contact network.
    budget : int
        Number of nodes to immunize.
    centrality_df : pd.DataFrame
        Pre-computed centrality table with a ``betweenness`` column.

    Returns
    -------
    G_immunized : nx.DiGraph
    targets : list
    """
    sorted_nodes = centrality_df.sort_values("betweenness", ascending=False).index.tolist()
    targets = _safe_nodes(G, sorted_nodes, budget)
    return _remove_nodes(G, targets), targets


def pagerank_immunization(
    G: nx.DiGraph,
    budget: int,
    centrality_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Immunize nodes with the highest *PageRank* score.

    Parameters
    ----------
    G : nx.DiGraph
        Contact network.
    budget : int
        Number of nodes to immunize.
    centrality_df : pd.DataFrame
        Pre-computed centrality table with a ``pagerank`` column.

    Returns
    -------
    G_immunized : nx.DiGraph
    targets : list
    """
    sorted_nodes = centrality_df.sort_values("pagerank", ascending=False).index.tolist()
    targets = _safe_nodes(G, sorted_nodes, budget)
    return _remove_nodes(G, targets), targets


def acquaintance_immunization(
    G: nx.DiGraph,
    budget: int,
    centrality_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Acquaintance (random-then-neighbor) immunization strategy.

    Select *budget* random nodes and immunize one random *neighbour* of each.
    This avoids requiring full degree knowledge while still targeting
    high-degree hubs (friendship paradox).

    Parameters
    ----------
    G : nx.DiGraph
        Contact network.
    budget : int
        Number of nodes to immunize.
    centrality_df : pd.DataFrame
        Pre-computed centrality table (unused, kept for API consistency).

    Returns
    -------
    G_immunized : nx.DiGraph
    targets : list
    """
    nodes = list(G.nodes())
    targets: List[Any] = []
    attempts = set()

    random.shuffle(nodes)
    for node in nodes:
        if len(targets) >= budget:
            break
        neighbours = list(G.successors(node)) if G.is_directed() else list(G.neighbors(node))
        if neighbours:
            candidate = random.choice(neighbours)
            if candidate not in attempts:
                targets.append(candidate)
                attempts.add(candidate)

    return _remove_nodes(G, targets), targets


def kcore_immunization(
    G: nx.DiGraph,
    budget: int,
    centrality_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Immunize nodes with the highest *k-core* shell number.

    Parameters
    ----------
    G : nx.DiGraph
        Contact network.
    budget : int
        Number of nodes to immunize.
    centrality_df : pd.DataFrame
        Pre-computed centrality table with a ``kcore`` column.

    Returns
    -------
    G_immunized : nx.DiGraph
    targets : list
    """
    sorted_nodes = centrality_df.sort_values("kcore", ascending=False).index.tolist()
    targets = _safe_nodes(G, sorted_nodes, budget)
    return _remove_nodes(G, targets), targets


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, Any] = {
    "random": random_immunization,
    "degree": degree_immunization,
    "betweenness": betweenness_immunization,
    "pagerank": pagerank_immunization,
    "acquaintance": acquaintance_immunization,
    "kcore": kcore_immunization,
}


def apply_strategy(
    name: str,
    G: nx.DiGraph,
    budget: int,
    centrality_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, List[Any]]:
    """Apply an immunization strategy by *name*.

    Parameters
    ----------
    name : str
        One of 'random', 'degree', 'betweenness', 'pagerank',
        'acquaintance', 'kcore'.
    G : nx.DiGraph
        Contact network.
    budget : int
        Number of nodes to immunize.
    centrality_df : pd.DataFrame
        Pre-computed centrality table.

    Returns
    -------
    G_immunized : nx.DiGraph
    targets : list

    Raises
    ------
    ValueError
        If *name* is not a recognised strategy.
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[name](G, budget, centrality_df)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_strategies(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
    seeds: List[Any],
    budget: int = 50,
    beta: float = 0.3,
    gamma: float = 0.1,
    steps: int = 100,
    runs: int = 10,
) -> pd.DataFrame:
    """Benchmark all six immunization strategies.

    For each strategy the function:
    1. Immunizes *budget* nodes.
    2. Runs *runs* independent SIR simulations on the immunized graph.
    3. Reports the mean and standard deviation of peak infected counts.

    Parameters
    ----------
    G : nx.DiGraph
        Original (non-immunized) contact network.
    centrality_df : pd.DataFrame
        Pre-computed centrality table.
    seeds : list
        Initial infected node IDs (seeds still present in G after immunization
        are used; if a seed was immunized it is excluded).
    budget : int
        Number of nodes to immunize per strategy.
    beta : float
        Infection probability per step.
    gamma : float
        Recovery probability per step.
    steps : int
        Maximum simulation steps per run.
    runs : int
        Number of Monte-Carlo repetitions per strategy.

    Returns
    -------
    pd.DataFrame
        Indexed by strategy name with columns:
        ``mean_infected``, ``std_infected``.
    """
    results = {}

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"[immunization] Benchmarking '{strategy_name}' …")
        G_imm, targets = strategy_fn(G, budget, centrality_df)

        # Remove any seeds that were immunized
        target_set = set(targets)
        effective_seeds = [s for s in seeds if s in G_imm and s not in target_set]
        if not effective_seeds:
            # Fall back: pick top-1 node by degree in immunized graph
            if G_imm.number_of_nodes() > 0:
                effective_seeds = [max(dict(G_imm.in_degree()).items(), key=lambda x: x[1])[0]]
            else:
                results[strategy_name] = {"mean_infected": 0, "std_infected": 0}
                continue

        peaks = []
        for _ in range(runs):
            history, _ = run_sir(G_imm, effective_seeds, beta=beta, gamma=gamma, steps=steps)
            peaks.append(peak_infected(history))

        results[strategy_name] = {
            "mean_infected": float(np.mean(peaks)),
            "std_infected": float(np.std(peaks)),
        }

    df = pd.DataFrame(results).T
    df.index.name = "strategy"
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.graph_builder import load_graph
    from src.centrality import compute_centralities

    G = load_graph()
    centrality_df = compute_centralities(G, k=min(200, G.number_of_nodes()))
    seeds = list(G.nodes())[:3]

    bench = benchmark_strategies(G, centrality_df, seeds, budget=30, runs=3)
    print(bench)
