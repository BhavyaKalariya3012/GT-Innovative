"""
app.py
------
Streamlit dashboard for the Misinformation Propagation & Immunization Simulator.

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import random

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ── Make the src package importable ──────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.graph_builder import load_graph
from src.centrality import compute_centralities
from src.epidemic_model import run_sir, run_sis, peak_infected
from src.immunization import apply_strategy, benchmark_strategies, STRATEGIES

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Misinformation Propagation & Immunization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦠",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark gradient background */
    .main {
        background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0d1b2a 100%);
        color: #e2e8f0;
    }
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0d1b2a 100%);
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
    }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 2rem;
        font-weight: 700;
    }
    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    /* Tabs / dividers */
    hr {
        border-color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; background: linear-gradient(90deg,#38bdf8,#818cf8,#f472b6);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   font-weight:800; letter-spacing:-1px;'>
            🦠 Misinformation Propagation &amp; Immunization Dashboard
        </h1>
        <p style='color:#94a3b8; font-size:1rem; margin-top:0.25rem;'>
            Simulate how false information spreads on social networks and evaluate immunization strategies.
        </p>
    </div>
    <hr style='border-color:#1e293b; margin-bottom:1.5rem;'/>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")
    st.markdown("---")

    model_type = st.radio(
        "Epidemic Model",
        options=["SIR", "SIS"],
        horizontal=True,
        key="model_type",
    )

    st.markdown("### 📡 Transmission Parameters")
    beta = st.slider(
        "β — Infection probability",
        min_value=0.01,
        max_value=0.90,
        value=0.30,
        step=0.01,
        help="Probability of transmission per edge per time step.",
    )

    if model_type == "SIR":
        gamma = st.slider(
            "γ — Recovery probability",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            help="Probability of recovery per infected node per step (SIR only).",
        )
    else:
        gamma = st.slider(
            "δ — Return-to-susceptible probability",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            help="Probability of recovery back to susceptible (SIS only).",
        )

    st.markdown("### 🛡️ Immunization Settings")
    strategy = st.selectbox(
        "Immunization Strategy",
        options=list(STRATEGIES.keys()),
        format_func=lambda s: s.capitalize(),
        key="strategy",
    )

    budget = st.slider(
        "Immunization Budget (nodes)",
        min_value=1,
        max_value=200,
        value=50,
        step=1,
        help="Number of nodes to immunize.",
    )

    st.markdown("---")
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b;'>Data: SNAP Twitter dataset or synthetic BA graph fallback.</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cache expensive computations
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_graph():
    """Load and cache the social graph."""
    return load_graph()


@st.cache_data(show_spinner=False)
def get_centralities(_G):
    """Compute and cache centrality metrics.

    The leading underscore tells Streamlit not to hash the graph argument.
    """
    k = min(500, _G.number_of_nodes())
    return compute_centralities(_G, k=k)


# ─────────────────────────────────────────────────────────────────────────────
# Build Pyvis network visualisation
# ─────────────────────────────────────────────────────────────────────────────

def build_pyvis_html(
    G: nx.DiGraph,
    final_state_baseline: dict,
    immunized_nodes: list,
    max_nodes: int = 300,
) -> str:
    """Render a Pyvis network coloured by simulation state.

    Node colours:
    - 🟢 Green  : immunized
    - 🔴 Red    : infected (I)
    - 🔵 Blue   : recovered (R)
    - ⚫ Gray   : susceptible (S)

    Parameters
    ----------
    G : nx.DiGraph
        Full social graph.
    final_state_baseline : dict
        Node → state mapping from the baseline simulation.
    immunized_nodes : list
        Nodes that were immunized.
    max_nodes : int
        Maximum number of nodes to render (for performance).

    Returns
    -------
    str
        HTML string of the Pyvis network.
    """
    immunized_set = set(immunized_nodes)

    # Sample nodes for visualisation
    all_nodes = list(G.nodes())
    if len(all_nodes) > max_nodes:
        sample_nodes = set(random.sample(all_nodes, max_nodes))
    else:
        sample_nodes = set(all_nodes)

    subG = G.subgraph(sample_nodes)

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0d1117",
        font_color="#e2e8f0",
        directed=True,
    )
    net.repulsion(node_distance=80, spring_length=100)

    colour_map = {
        "S": "#64748b",   # gray
        "I": "#ef4444",   # red
        "R": "#3b82f6",   # blue
        "IMM": "#22c55e", # green
    }

    for node in subG.nodes():
        if node in immunized_set:
            colour = colour_map["IMM"]
            title = f"Node {node} — Immunized"
            size = 12
        else:
            s = final_state_baseline.get(node, "S")
            colour = colour_map.get(s, colour_map["S"])
            title = f"Node {node} — {s}"
            size = 8

        net.add_node(
            str(node),
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


# ─────────────────────────────────────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.6)",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", zerolinecolor="#1e293b"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)


def plot_epidemic_curves(
    history_baseline: list,
    history_immunized: list,
    model: str,
) -> go.Figure:
    """Build a Plotly line chart comparing baseline vs immunized infected curves.

    Parameters
    ----------
    history_baseline : list of dict
        Simulation history without immunization.
    history_immunized : list of dict
        Simulation history with immunization applied.
    model : str
        'SIR' or 'SIS'.

    Returns
    -------
    go.Figure
    """
    steps_b = list(range(len(history_baseline)))
    steps_i = list(range(len(history_immunized)))

    infected_b = [h["I"] for h in history_baseline]
    infected_i = [h["I"] for h in history_immunized]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps_b,
        y=infected_b,
        mode="lines",
        name="Baseline (no immunization)",
        line=dict(color="#ef4444", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.12)",
    ))

    fig.add_trace(go.Scatter(
        x=steps_i,
        y=infected_i,
        mode="lines",
        name=f"Immunized ({strategy.capitalize()})",
        line=dict(color="#22c55e", width=2.5, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.10)",
    ))

    fig.update_layout(
        title=dict(text=f"📈 {model} Epidemic Curve — Infected Over Time", font=dict(size=16, color="#f1f5f9")),
        xaxis_title="Time Step",
        yaxis_title="Number of Infected Nodes",
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_strategy_bars(bench_df: pd.DataFrame) -> go.Figure:
    """Build a Plotly bar chart with error bars for strategy comparison.

    Parameters
    ----------
    bench_df : pd.DataFrame
        Output of :func:`benchmark_strategies` with columns
        ``mean_infected`` and ``std_infected``.

    Returns
    -------
    go.Figure
    """
    strategies = bench_df.index.tolist()
    means = bench_df["mean_infected"].tolist()
    stds = bench_df["std_infected"].tolist()

    colours = [
        "#3b82f6", "#8b5cf6", "#ec4899",
        "#f59e0b", "#10b981", "#06b6d4",
    ]

    fig = go.Figure(
        go.Bar(
            x=strategies,
            y=means,
            error_y=dict(type="data", array=stds, visible=True, color="#94a3b8"),
            marker=dict(
                color=colours[: len(strategies)],
                line=dict(color="#1e293b", width=1),
            ),
            text=[f"{m:.1f}" for m in means],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
        )
    )

    fig.update_layout(
        title=dict(text="🛡️ Strategy Comparison — Mean Peak Infected", font=dict(size=16, color="#f1f5f9")),
        xaxis_title="Immunization Strategy",
        yaxis_title="Mean Peak Infected Nodes",
        **_PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Data Explorer  (always visible — loads graph & centralities on startup)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("🔍 Data Explorer — Browse Graph & Centrality Data", expanded=False):

    with st.spinner("Loading graph data …"):
        _G_exp  = get_graph()
        _cdf    = get_centralities(_G_exp)

    # ── Graph Statistics ──────────────────────────────────────────────────
    st.markdown("### 📐 Graph Statistics")
    _und = _G_exp.to_undirected()
    _stat_cols = st.columns(4)
    _stat_cols[0].metric("Nodes", f"{_G_exp.number_of_nodes():,}")
    _stat_cols[1].metric("Edges", f"{_G_exp.number_of_edges():,}")
    _stat_cols[2].metric("Density", f"{nx.density(_G_exp):.5f}")
    _stat_cols[3].metric("Avg Clustering", f"{nx.average_clustering(_und):.4f}")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Centrality Table",
        "🎯 Top Nodes to Target",
        "📈 Degree Distribution",
    ])

    # Tab 1 — full centrality DataFrame, searchable & sortable
    with tab1:
        st.markdown(
            "<small style='color:#64748b;'>Sort any column by clicking its header. "
            "All six centrality scores + composite ranking.</small>",
            unsafe_allow_html=True,
        )
        _top_n = st.slider(
            "Rows to display", min_value=10, max_value=min(500, len(_cdf)),
            value=50, step=10, key="explorer_rows"
        )
        _display_df = _cdf.head(_top_n).reset_index()
        _display_df.columns = [c.capitalize() for c in _display_df.columns]
        _display_df = _display_df.round(6)
        st.dataframe(_display_df, use_container_width=True, height=380)

        # Download button
        _csv = _cdf.reset_index().round(6).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download full centrality CSV",
            data=_csv,
            file_name="centrality_scores.csv",
            mime="text/csv",
        )

    # Tab 2 — top N nodes recommended for each strategy
    with tab2:
        st.markdown("**Top 20 nodes recommended per immunization metric** (sorted descending):")
        _t2_cols = st.columns(3)
        _metrics  = [("Degree", "degree"), ("Betweenness", "betweenness"), ("PageRank", "pagerank"),
                     ("Eigenvector", "eigenvector"), ("K-Core", "kcore"), ("Composite", "composite")]
        for idx, (label, col) in enumerate(_metrics):
            with _t2_cols[idx % 3]:
                st.markdown(f"**{label}**")
                _top20 = _cdf.sort_values(col, ascending=False).head(20)[[col]].reset_index()
                _top20.columns = ["Node ID", label]
                _top20[label] = _top20[label].round(5)
                st.dataframe(_top20, use_container_width=True, height=260)

    # Tab 3 — degree distribution log-log Plotly chart
    with tab3:
        import numpy as np
        _degrees = [d for _, d in _G_exp.in_degree()]
        _deg_counts = pd.Series(_degrees).value_counts().sort_index()
        _fig_deg = go.Figure(go.Scatter(
            x=_deg_counts.index.tolist(),
            y=_deg_counts.values.tolist(),
            mode="markers",
            marker=dict(color="#38bdf8", size=5, opacity=0.7),
            name="In-degree frequency",
        ))
        _fig_deg.update_layout(
            title=dict(text="In-Degree Distribution (log-log scale)", font=dict(size=15, color="#f1f5f9")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.6)",
            font=dict(color="#e2e8f0", family="Inter, sans-serif"),
            xaxis=dict(title="In-Degree (k)", type="log", gridcolor="#1e293b", linecolor="#334155"),
            yaxis=dict(title="Count", type="log", gridcolor="#1e293b", linecolor="#334155"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(_fig_deg, use_container_width=True)
        st.markdown(
            "<small style='color:#64748b;'>A straight line on a log-log plot indicates a power-law "
            "(scale-free) degree distribution — typical of real social networks.</small>",
            unsafe_allow_html=True,
        )

st.markdown("<hr style='border-color:#1e293b; margin:1.5rem 0;'/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main simulation logic (on button press)
# ─────────────────────────────────────────────────────────────────────────────

if run_button:
    with st.spinner("⚙️ Loading graph and computing centralities …"):
        G = get_graph()
        centrality_df = get_centralities(G)

    # Pick top-3 nodes by degree as seeds
    seeds = centrality_df.sort_values("degree", ascending=False).head(3).index.tolist()
    seeds = [s for s in seeds if s in G]

    with st.spinner(f"🛡️ Applying **{strategy}** immunization strategy …"):
        G_imm, immunized_nodes = apply_strategy(strategy, G, budget, centrality_df)

    # Effective seeds for immunized graph
    imm_set = set(immunized_nodes)
    seeds_imm = [s for s in seeds if s in G_imm and s not in imm_set]
    if not seeds_imm and G_imm.number_of_nodes() > 0:
        seeds_imm = [list(G_imm.nodes())[0]]

    with st.spinner("🔬 Running simulations …"):
        if model_type == "SIR":
            history_baseline, final_state_b = run_sir(G, seeds, beta=beta, gamma=gamma, steps=100)
            history_immunized, final_state_i = run_sir(G_imm, seeds_imm, beta=beta, gamma=gamma, steps=100)
        else:
            history_baseline, final_state_b = run_sis(G, seeds, beta=beta, delta=gamma, steps=100)
            history_immunized, final_state_i = run_sis(G_imm, seeds_imm, beta=beta, delta=gamma, steps=100)

    with st.spinner("📊 Benchmarking all strategies (this may take a moment) …"):
        bench_df = benchmark_strategies(
            G, centrality_df, seeds,
            budget=budget, beta=beta, gamma=gamma,
            steps=60, runs=10,
        )

    # ── Layout ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([6, 4], gap="large")

    # ── Left column ───────────────────────────────────────────────────────
    with col_left:
        st.plotly_chart(
            plot_epidemic_curves(history_baseline, history_immunized, model_type),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_strategy_bars(bench_df),
            use_container_width=True,
        )

    # ── Right column ──────────────────────────────────────────────────────
    with col_right:
        # Metrics
        peak_b = peak_infected(history_baseline)
        peak_i = peak_infected(history_immunized)
        reduction = ((peak_b - peak_i) / max(peak_b, 1)) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Peak (Baseline)", f"{peak_b:,}", delta=None)
        m2.metric("Peak (Immunized)", f"{peak_i:,}", delta=f"-{peak_b - peak_i:,}", delta_color="inverse")
        m3.metric("Spread Reduction", f"{reduction:.1f}%", delta=None)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Pyvis network
        st.markdown(
            "<h4 style='color:#94a3b8; font-size:0.95rem; margin-bottom:0;'>"
            "🕸️ Network Visualisation (first 300 nodes)"
            "</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<small style='color:#475569;'>"
            "🟢 Immunized &nbsp; 🔴 Infected &nbsp; 🔵 Recovered &nbsp; ⚫ Susceptible"
            "</small>",
            unsafe_allow_html=True,
        )

        with st.spinner("Rendering network …"):
            pyvis_html = build_pyvis_html(G, final_state_b, immunized_nodes, max_nodes=300)
            components.html(pyvis_html, height=520, scrolling=False)

        # Strategy benchmark table
        st.markdown("#### 📋 Strategy Benchmark Results")
        bench_display = bench_df.copy()
        bench_display.columns = ["Mean Peak Infected", "Std Dev"]
        bench_display = bench_display.round(2)
        bench_display.index = bench_display.index.str.capitalize()
        st.dataframe(bench_display, use_container_width=True)

else:
    # ── Landing placeholder ────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding:4rem 2rem;
                    background:linear-gradient(135deg,#1e293b20,#0f172a40);
                    border:1px dashed #334155; border-radius:16px; margin-top:1rem;'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🦠</div>
            <h2 style='color:#94a3b8; font-weight:600;'>Ready to Simulate</h2>
            <p style='color:#64748b; max-width:520px; margin:0 auto;'>
                Adjust the parameters in the sidebar, then press
                <strong style='color:#38bdf8;'>Run Simulation</strong>
                to model misinformation propagation and evaluate immunization strategies
                on the Twitter social graph (or a synthetic fallback graph).
            </p>
            <br/>
            <div style='display:flex; justify-content:center; gap:2rem; flex-wrap:wrap; margin-top:1rem;'>
                <div style='padding:1rem 1.5rem; background:#1e293b; border-radius:10px;
                            border:1px solid #334155; min-width:140px;'>
                    <div style='font-size:1.5rem;'>📡</div>
                    <p style='color:#94a3b8; font-size:0.85rem; margin:0.25rem 0 0;'>SIR / SIS Models</p>
                </div>
                <div style='padding:1rem 1.5rem; background:#1e293b; border-radius:10px;
                            border:1px solid #334155; min-width:140px;'>
                    <div style='font-size:1.5rem;'>🛡️</div>
                    <p style='color:#94a3b8; font-size:0.85rem; margin:0.25rem 0 0;'>6 Immunization Strategies</p>
                </div>
                <div style='padding:1rem 1.5rem; background:#1e293b; border-radius:10px;
                            border:1px solid #334155; min-width:140px;'>
                    <div style='font-size:1.5rem;'>📊</div>
                    <p style='color:#94a3b8; font-size:0.85rem; margin:0.25rem 0 0;'>Live Plotly Charts</p>
                </div>
                <div style='padding:1rem 1.5rem; background:#1e293b; border-radius:10px;
                            border:1px solid #334155; min-width:140px;'>
                    <div style='font-size:1.5rem;'>🕸️</div>
                    <p style='color:#94a3b8; font-size:0.85rem; margin:0.25rem 0 0;'>Interactive Network</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
