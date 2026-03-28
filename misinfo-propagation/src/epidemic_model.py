"""
epidemic_model.py
-----------------
Pure-Python (no NDlib) implementations of the SIR and SIS epidemic models
for simulating misinformation propagation on a NetworkX graph.
"""

import random
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# SIR Model
# ---------------------------------------------------------------------------

def run_sir(
    G,
    seeds: List[Any],
    beta: float = 0.3,
    gamma: float = 0.1,
    steps: int = 100,
) -> tuple:
    """Simulate SIR (Susceptible-Infected-Recovered) dynamics on *G*.

    At each discrete time step every **Infected** node independently
    attempts to infect each of its **Susceptible** neighbours with
    probability *beta*, and independently recovers with probability *gamma*.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Contact network.  For directed graphs outgoing neighbours are used.
    seeds : list
        Initial set of infected node IDs.
    beta : float
        Infection (transmission) probability per edge per step. Range (0, 1].
    gamma : float
        Recovery probability per infected node per step. Range (0, 1].
    steps : int
        Maximum number of simulation steps.

    Returns
    -------
    history : list of dict
        One entry per step with keys ``S``, ``I``, ``R`` (counts).
    final_state : dict
        Mapping ``node -> state`` ('S', 'I', or 'R') at the end of the run.
    """
    nodes = list(G.nodes())
    seed_set = set(seeds)

    # Initialise states
    state: Dict[Any, str] = {}
    for node in nodes:
        state[node] = "I" if node in seed_set else "S"

    history: List[Dict[str, int]] = []

    for _ in range(steps):
        new_state = state.copy()

        for node in nodes:
            if state[node] == "I":
                # Attempt to infect susceptible neighbours
                for neighbour in G.successors(node) if G.is_directed() else G.neighbors(node):
                    if state[neighbour] == "S":
                        if random.random() < beta:
                            new_state[neighbour] = "I"
                # Recovery
                if random.random() < gamma:
                    new_state[node] = "R"

        state = new_state

        counts = _count_states_sir(state)
        history.append(counts)

        # Early termination: no infected nodes remain
        if counts["I"] == 0:
            break

    return history, state


def _count_states_sir(state: Dict[Any, str]) -> Dict[str, int]:
    """Count nodes in each SIR compartment.

    Parameters
    ----------
    state : dict
        Mapping node -> 'S', 'I', or 'R'.

    Returns
    -------
    dict
        ``{"S": int, "I": int, "R": int}``
    """
    counts = {"S": 0, "I": 0, "R": 0}
    for s in state.values():
        counts[s] += 1
    return counts


# ---------------------------------------------------------------------------
# SIS Model
# ---------------------------------------------------------------------------

def run_sis(
    G,
    seeds: List[Any],
    beta: float = 0.3,
    delta: float = 0.1,
    steps: int = 100,
) -> tuple:
    """Simulate SIS (Susceptible-Infected-Susceptible) dynamics on *G*.

    Recovered nodes return to the **Susceptible** state, allowing cycles
    of re-infection.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Contact network.
    seeds : list
        Initial set of infected node IDs.
    beta : float
        Infection probability per edge per step. Range (0, 1].
    delta : float
        Recovery (return-to-susceptible) probability per step. Range (0, 1].
    steps : int
        Maximum number of simulation steps.

    Returns
    -------
    history : list of dict
        One entry per step with keys ``S``, ``I`` (counts).
    final_state : dict
        Mapping ``node -> state`` ('S' or 'I') at the end of the run.
    """
    nodes = list(G.nodes())
    seed_set = set(seeds)

    state: Dict[Any, str] = {}
    for node in nodes:
        state[node] = "I" if node in seed_set else "S"

    history: List[Dict[str, int]] = []

    for _ in range(steps):
        new_state = state.copy()

        for node in nodes:
            if state[node] == "I":
                # Attempt to infect susceptible neighbours
                for neighbour in G.successors(node) if G.is_directed() else G.neighbors(node):
                    if state[neighbour] == "S":
                        if random.random() < beta:
                            new_state[neighbour] = "I"
                # Recovery back to susceptible
                if random.random() < delta:
                    new_state[node] = "S"

        state = new_state

        counts = _count_states_sis(state)
        history.append(counts)

        if counts["I"] == 0:
            break

    return history, state


def _count_states_sis(state: Dict[Any, str]) -> Dict[str, int]:
    """Count nodes in each SIS compartment.

    Parameters
    ----------
    state : dict
        Mapping node -> 'S' or 'I'.

    Returns
    -------
    dict
        ``{"S": int, "I": int}``
    """
    counts = {"S": 0, "I": 0}
    for s in state.values():
        counts[s] += 1
    return counts


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def peak_infected(history: List[Dict[str, int]]) -> int:
    """Return the maximum infected count from a simulation history.

    Parameters
    ----------
    history : list of dict
        Output from :func:`run_sir` or :func:`run_sis`.

    Returns
    -------
    int
        Peak number of simultaneously infected nodes.
    """
    return max(step["I"] for step in history) if history else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.graph_builder import load_graph

    G = load_graph()
    seeds = list(G.nodes())[:5]

    print("=== SIR ===")
    history_sir, final_sir = run_sir(G, seeds, beta=0.3, gamma=0.1, steps=50)
    print(f"Steps: {len(history_sir)}, Peak I: {peak_infected(history_sir)}")

    print("=== SIS ===")
    history_sis, final_sis = run_sis(G, seeds, beta=0.3, delta=0.1, steps=50)
    print(f"Steps: {len(history_sis)}, Peak I: {peak_infected(history_sis)}")
