"""
graph_builder.py
----------------
Loads and preprocesses the SNAP Twitter dataset (or falls back to a
synthetic Barabasi-Albert graph).  Returns a directed NetworkX graph
representing the largest weakly-connected component.
"""

import os
import sys
import networkx as nx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_graph(data_path: str = None) -> nx.DiGraph:
    """Load the Twitter social graph from a SNAP edge-list file.

    Parameters
    ----------
    data_path : str, optional
        Absolute or relative path to ``twitter_combined.txt``.
        If *None* the function tries ``data/twitter_combined.txt``
        relative to this file's parent directory.

    Returns
    -------
    nx.DiGraph
        Pre-processed directed graph (giant weakly-connected component,
        self-loops removed).
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "twitter_combined.txt")

    if os.path.exists(data_path):
        print(f"[graph_builder] Loading SNAP Twitter dataset from: {data_path}")
        G_raw = nx.read_edgelist(
            data_path,
            create_using=nx.DiGraph(),
            nodetype=int,
        )
    else:
        print(
            "[graph_builder] twitter_combined.txt not found – "
            "generating synthetic Barabasi-Albert graph (n=1000, m=3)."
        )
        G_raw = _synthetic_graph(n=1000, m=3)

    G = _preprocess(G_raw)
    _print_stats(G)
    return G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _synthetic_graph(n: int = 1000, m: int = 3) -> nx.DiGraph:
    """Return a directed Barabasi-Albert random graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges to attach from a new node to existing nodes.

    Returns
    -------
    nx.DiGraph
    """
    undirected = nx.barabasi_albert_graph(n, m, seed=42)
    return undirected.to_directed()


def _preprocess(G: nx.DiGraph) -> nx.DiGraph:
    """Extract the giant weakly-connected component and remove self-loops.

    Parameters
    ----------
    G : nx.DiGraph
        Raw input graph.

    Returns
    -------
    nx.DiGraph
        Cleaned graph.
    """
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Keep only the largest weakly-connected component
    components = list(nx.weakly_connected_components(G))
    if not components:
        raise ValueError("Graph has no weakly-connected components.")

    giant = max(components, key=len)
    G_giant = G.subgraph(giant).copy()

    return G_giant


def _print_stats(G: nx.DiGraph) -> None:
    """Print basic graph statistics to stdout.

    Parameters
    ----------
    G : nx.DiGraph
        Graph to analyse.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)

    # Clustering is defined on the underlying undirected graph
    undirected = G.to_undirected()
    avg_clustering = nx.average_clustering(undirected)

    print(f"[graph_builder] Nodes          : {n:,}")
    print(f"[graph_builder] Edges          : {e:,}")
    print(f"[graph_builder] Density        : {density:.6f}")
    print(f"[graph_builder] Avg clustering : {avg_clustering:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point (optional direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    G = load_graph(path)
    print(f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
