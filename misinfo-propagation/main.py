"""
main.py
-------
CLI orchestrator for the Intelligent Misinformation Propagation System.

Runs all four features in sequence:
  1. Graph loading
  2. Influencer detection (centrality analysis)
  3. Misinformation classification (NLP / TF-IDF + LR)
  4. Epidemic simulation (SIR baseline)
  5. Containment strategy comparison
  6. Visualization (matplotlib PNG)
  7. Final summary report

Usage:
    python main.py [--nodes 5] [--beta 0.3] [--gamma 0.1] [--k 10] [--runs 5]
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make the src package importable ──────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.graph_builder  import load_graph
from src.centrality     import compute_centralities, top_influencers
from src.epidemic_model import run_sir, peak_infected
from src.detection      import (
    train_detector,
    generate_node_texts,
    classify_nodes,
    detection_summary,
)
from src.containment    import compare_containment, print_containment_report
from src.visualization  import save_matplotlib_png


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Intelligent Misinformation Propagation System — CLI Runner"
    )
    p.add_argument("--nodes",  type=int,   default=5,    help="Number of seed (initially infected) nodes")
    p.add_argument("--beta",   type=float, default=0.30, help="SIR infection probability β")
    p.add_argument("--gamma",  type=float, default=0.10, help="SIR recovery probability γ")
    p.add_argument("--k",      type=int,   default=10,   help="Top-K influencers / containment budget")
    p.add_argument("--runs",   type=int,   default=5,    help="Monte-Carlo runs for containment benchmark")
    p.add_argument("--steps",  type=int,   default=100,  help="Max SIR simulation steps")
    p.add_argument("--no-viz", action="store_true",      help="Skip matplotlib visualization")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 60
    print("\n" + "━" * width)
    print(f"  {title}")
    print("━" * width)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 60)
    print("  🦠  Intelligent Misinformation Propagation System")
    print("=" * 60)
    print(f"  β={args.beta}  γ={args.gamma}  seeds={args.nodes}  "
          f"k={args.k}  runs={args.runs}")

    # ──────────────────────────────────────────────────────────────
    # STEP 1 — Load Graph
    # ──────────────────────────────────────────────────────────────
    _banner("STEP 1 — Loading Social Network Graph")
    G = load_graph()
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  Graph ready: {n_nodes:,} nodes | {n_edges:,} edges")

    # ──────────────────────────────────────────────────────────────
    # STEP 2 — Centrality & Influencer Detection
    # ──────────────────────────────────────────────────────────────
    _banner("STEP 2 — Influencer Detection (Centrality Analysis)")
    k_centrality = min(200, n_nodes)
    centrality_df = compute_centralities(G, k=k_centrality)
    influencer_df = top_influencers(centrality_df, k=args.k)
    influencer_nodes = influencer_df.index.tolist()

    # ──────────────────────────────────────────────────────────────
    # STEP 3 — Misinformation Detection (NLP classifier)
    # ──────────────────────────────────────────────────────────────
    _banner("STEP 3 — Misinformation Detection (TF-IDF + Logistic Regression)")
    pipeline     = train_detector(verbose=True)
    node_texts   = generate_node_texts(G, fake_fraction=0.35)
    node_labels  = classify_nodes(G, pipeline, node_texts)
    det_summary  = detection_summary(node_labels)

    # Identify which influencer accounts are also labelled FAKE
    fake_influencers = [n for n in influencer_nodes if node_labels.get(n) == "FAKE"]
    print(f"  ⚠️  Fake-labelled influencers : "
          f"{len(fake_influencers)} / {len(influencer_nodes)}")

    # ──────────────────────────────────────────────────────────────
    # STEP 4 — Baseline SIR Simulation
    # ──────────────────────────────────────────────────────────────
    _banner("STEP 4 — Baseline SIR Epidemic Simulation")
    all_nodes = list(G.nodes())
    seeds     = all_nodes[: args.nodes]
    print(f"  Seed nodes : {seeds}")

    history_baseline, final_state = run_sir(
        G, seeds,
        beta=args.beta,
        gamma=args.gamma,
        steps=args.steps,
    )
    peak_baseline = peak_infected(history_baseline)
    total_recovered = sum(1 for s in final_state.values() if s == "R")
    total_infected  = sum(1 for s in final_state.values() if s == "I")

    print(f"\n  🔴 Peak infected (baseline)   : {peak_baseline:,}")
    print(f"  🔵 Total recovered (final)    : {total_recovered:,}")
    print(f"  🔴 Still infected (final)     : {total_infected:,}")
    print(f"  📈 Simulation steps run       : {len(history_baseline)}")

    # ──────────────────────────────────────────────────────────────
    # STEP 5 — Containment Strategy Comparison
    # ──────────────────────────────────────────────────────────────
    _banner("STEP 5 — Containment Strategy Comparison")
    containment_df = compare_containment(
        G,
        centrality_df,
        seeds,
        beta=args.beta,
        gamma=args.gamma,
        k=args.k,
        runs=args.runs,
        steps=args.steps,
    )
    print_containment_report(containment_df)

    best_strategy = (
        containment_df
        .loc[containment_df.index != "baseline", "reduction_pct"]
        .idxmax()
    )
    best_reduction = containment_df.loc[best_strategy, "reduction_pct"]
    best_mean      = containment_df.loc[best_strategy, "mean_peak"]

    # ──────────────────────────────────────────────────────────────
    # STEP 6 — Visualization
    # ──────────────────────────────────────────────────────────────
    if not args.no_viz:
        _banner("STEP 6 — Network Visualization (Matplotlib PNG)")
        png_path = save_matplotlib_png(
            G,
            final_state,
            influencers=influencer_nodes,
            filename="network_final_state.png",
        )
        if png_path:
            print(f"  ✅ Saved → {png_path}")
        else:
            print("  ⚠️  Matplotlib not available — visualization skipped.")
    else:
        png_path = None
        print("\n  [--no-viz flag set — skipping visualization]")

    # ──────────────────────────────────────────────────────────────
    # STEP 7 — Final Summary Report
    # ──────────────────────────────────────────────────────────────
    _banner("📊 FINAL SUMMARY REPORT")

    print(f"""
  Graph
  ─────────────────────────────────────────────
  Nodes                    : {n_nodes:>10,}
  Edges                    : {n_edges:>10,}

  Influencer Detection
  ─────────────────────────────────────────────
  Top-{args.k} influencers found   : {len(influencer_nodes):>10}
  Influencers labelled FAKE : {len(fake_influencers):>10}

  Misinformation Detection
  ─────────────────────────────────────────────
  Total nodes classified   : {det_summary['total']:>10,}
  FAKE nodes               : {det_summary['fake']:>10,}  ({det_summary['fake_pct']:.1f}%)
  REAL nodes               : {det_summary['real']:>10,}  ({det_summary['real_pct']:.1f}%)

  Epidemic Spread (Baseline SIR)
  ─────────────────────────────────────────────
  Peak infected (baseline) : {peak_baseline:>10,}
  Final recovered          : {total_recovered:>10,}
  Final still infected     : {total_infected:>10,}

  Best Containment Strategy
  ─────────────────────────────────────────────
  Strategy                 : {best_strategy:>10}
  Mean peak (contained)    : {best_mean:>10.1f}
  Spread reduction         : {best_reduction:>9.1f}%

  Visualization
  ─────────────────────────────────────────────
  PNG saved to             : {png_path or 'N/A (matplotlib not installed)'}
""")
    print("=" * 60)
    print("  ✅  Run complete.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
