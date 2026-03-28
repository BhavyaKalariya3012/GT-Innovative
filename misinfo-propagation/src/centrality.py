"""
centrality.py
-------------
Computes multiple centrality metrics for nodes in a directed graph and
returns a consolidated pandas DataFrame with a composite ranking score.
"""

import networkx as nx
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_centralities(G: nx.DiGraph, k: int = 500) -> pd.DataFrame:
    """Compute six centrality measures for every node in *G*.

    The following measures are computed:
    - **degree**       : in-degree centrality (normalised)
    - **betweenness**  : betweenness centrality (approximate, sampled from *k* nodes)
    - **eigenvector**  : eigenvector centrality of the underlying undirected graph
    - **pagerank**     : PageRank (damping = 0.85)
    - **closeness**    : closeness centrality (undirected)
    - **kcore**        : k-core number (degeneracy shell index)

    A **composite** column is added as the arithmetic mean of the
    (min-max normalised) degree, betweenness, eigenvector, and PageRank scores.

    Parameters
    ----------
    G : nx.DiGraph
        Input directed graph.
    k : int, optional
        Number of pivot nodes used for approximate betweenness computation.
        Default is 500.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by node ID, sorted descending by *composite* score.
        Columns: ``degree``, ``betweenness``, ``eigenvector``, ``pagerank``,
        ``closeness``, ``kcore``, ``composite``.
    """
    print("[centrality] Computing degree centrality …")
    degree_cent = nx.in_degree_centrality(G)

    print(f"[centrality] Computing betweenness centrality (k={k}) …")
    betweenness_cent = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)

    print("[centrality] Computing eigenvector centrality …")
    undirected = G.to_undirected()
    try:
        eigen_cent = nx.eigenvector_centrality(undirected, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        # Fall back to a uniform distribution if power iteration doesn't converge
        eigen_cent = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}

    print("[centrality] Computing PageRank …")
    pagerank_cent = nx.pagerank(G, alpha=0.85)

    print("[centrality] Computing closeness centrality …")
    closeness_cent = nx.closeness_centrality(undirected)

    print("[centrality] Computing k-core …")
    kcore_cent = nx.core_number(undirected)

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "degree": degree_cent,
            "betweenness": betweenness_cent,
            "eigenvector": eigen_cent,
            "pagerank": pagerank_cent,
            "closeness": closeness_cent,
            "kcore": kcore_cent,
        }
    )
    df.index.name = "node"

    # Composite score: mean of min-max normalised degree, betweenness,
    # eigenvector, and pagerank
    df["composite"] = _composite_score(df, ["degree", "betweenness", "eigenvector", "pagerank"])
    df.sort_values("composite", ascending=False, inplace=True)

    print(f"[centrality] Done – computed centralities for {len(df):,} nodes.")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _minmax_normalise(series: pd.Series) -> pd.Series:
    """Return a min-max normalised copy of *series* (values in [0, 1]).

    Parameters
    ----------
    series : pd.Series
        Input data.

    Returns
    -------
    pd.Series
        Normalised series; returns all-zeros if max == min.
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _composite_score(df: pd.DataFrame, columns: list) -> pd.Series:
    """Compute the mean of min-max normalised *columns* from *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    columns : list of str
        Column names to include in the composite score.

    Returns
    -------
    pd.Series
        Composite score per node.
    """
    normalised = pd.concat(
        [_minmax_normalise(df[col]) for col in columns], axis=1
    )
    return normalised.mean(axis=1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.graph_builder import load_graph

    G = load_graph()
    df = compute_centralities(G, k=min(500, G.number_of_nodes()))
    print(df.head(10))
