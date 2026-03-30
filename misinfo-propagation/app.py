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
from src.detection import train_detector, generate_node_texts, classify_nodes, detection_summary

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
# R₀ and Herd Immunity Threshold helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_r0_hit(beta: float, gamma: float, G: nx.DiGraph):
    """Compute Basic Reproduction Number R₀ and Herd Immunity Threshold (HIT).

    Uses the mean-field approximation: R₀ = β × <k> / γ where <k> is the
    average degree of the contact network.

    Parameters
    ----------
    beta : float
        Transmission probability per edge per step.
    gamma : float
        Recovery probability per step.
    G : nx.DiGraph
        Contact network.

    Returns
    -------
    r0 : float
        Basic reproduction number.
    hit : float
        Herd immunity threshold as a percentage (0–100).
    """
    n = G.number_of_nodes()
    avg_degree = sum(d for _, d in G.degree()) / n if n > 0 else 0.0
    r0 = (beta * avg_degree / gamma) if gamma > 0 else float("inf")
    hit = max(0.0, (1.0 - 1.0 / r0) * 100.0) if r0 > 1 else 0.0
    return round(r0, 2), round(hit, 1)


def plot_r0_gauge(r0: float) -> go.Figure:
    """Render a Plotly gauge for the Basic Reproduction Number.

    Parameters
    ----------
    r0 : float
        Basic reproduction number.

    Returns
    -------
    go.Figure
    """
    max_val = max(15.0, r0 * 1.5)
    colour  = "#ef4444" if r0 > 3 else ("#f59e0b" if r0 > 1 else "#22c55e")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r0,
        title=dict(text="Basic Reproduction Number R₀", font=dict(color="#94a3b8", size=13)),
        number=dict(font=dict(color="#f1f5f9", size=40), suffix=""),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="#94a3b8",
                      tickfont=dict(color="#94a3b8"), nticks=6),
            bar=dict(color=colour, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,       1],       color="rgba(34,197,94,0.12)"),
                dict(range=[1,       3],       color="rgba(251,191,36,0.12)"),
                dict(range=[3, max_val],       color="rgba(239,68,68,0.12)"),
            ],
            threshold=dict(
                line=dict(color="#f8fafc", width=3),
                thickness=0.75, value=1,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        height=260,
        margin=dict(l=20, r=20, t=55, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Temporal spread animation helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_animated_spread(history: list, model: str, label: str = "") -> go.Figure:
    """Build an animated Plotly bar chart that plays through each simulation step.

    Parameters
    ----------
    history : list of dict
        Per-step counts from :func:`run_sir` or :func:`run_sis`.
    model : str
        'SIR' or 'SIS' — determines which compartments to show.
    label : str, optional
        Extra label appended to the chart title (e.g. 'Baseline').

    Returns
    -------
    go.Figure
        Plotly figure with animation frames and play/pause controls.
    """
    keys   = ["S", "I", "R"] if model == "SIR" else ["S", "I"]
    clr    = {"S": "#64748b", "I": "#ef4444", "R": "#3b82f6"}
    names  = {"S": "Susceptible", "I": "Infected", "R": "Recovered"}
    n      = len(history)

    def _bar(h):
        return go.Bar(
            x=[names.get(k, k) for k in keys],
            y=[h.get(k, 0) for k in keys],
            marker_color=[clr.get(k, "#64748b") for k in keys],
            marker_line=dict(color="#0d1117", width=1),
            text=[f"{h.get(k, 0):,}" for k in keys],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
        )

    frames = [
        go.Frame(
            data=[_bar(history[i])],
            name=str(i),
            layout=go.Layout(title_text=f"⏱ Step {i + 1} / {n}  {label}"),
        )
        for i in range(n)
    ]

    fig = go.Figure(data=[_bar(history[0])], frames=frames)

    max_y = max(max(h.get(k, 0) for k in keys) for h in history) * 1.15

    fig.update_layout(
        title=dict(
            text=f"⏱ Step 1 / {n}  {label}",
            font=dict(size=14, color="#f1f5f9"),
        ),
        xaxis=dict(title="Compartment", gridcolor="#1e293b", linecolor="#334155"),
        yaxis=dict(title="Nodes", gridcolor="#1e293b", linecolor="#334155",
                   range=[0, max_y]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.6)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=60, b=80),
        height=380,
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.5, y=-0.18, xanchor="center",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
            bgcolor="#1e293b", bordercolor="#334155",
            font=dict(color="#e2e8f0"),
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Step: ", font=dict(color="#94a3b8")),
            pad=dict(b=5, t=45),
            x=0, len=1,
            steps=[dict(
                args=[[str(i)],
                      dict(frame=dict(duration=120, redraw=True), mode="immediate")],
                label=str(i + 1),
                method="animate",
            ) for i in range(n)],
            bgcolor="#1e293b", bordercolor="#334155",
            font=dict(color="#94a3b8", size=9),
            tickcolor="#334155",
        )],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Community detection and heatmap helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_communities(G: nx.DiGraph):
    """Detect communities using Greedy Modularity Optimisation.

    Parameters
    ----------
    G : nx.DiGraph
        Contact network (converted to undirected internally).

    Returns
    -------
    node_community : dict
        Mapping node → integer community ID.
    n_communities : int
        Total number of detected communities.
    """
    undirected = G.to_undirected()
    raw = list(nx.community.greedy_modularity_communities(undirected, weight=None))
    node_community = {}
    for cid, comm in enumerate(raw):
        for node in comm:
            node_community[node] = cid
    return node_community, len(raw)


def plot_community_heatmap(
    G: nx.DiGraph,
    final_state: dict,
    node_community: dict,
) -> go.Figure:
    """Stacked bar chart showing infection distribution per community.

    Parameters
    ----------
    G : nx.DiGraph
        Contact network.
    final_state : dict
        Node → state mapping from the simulation.
    node_community : dict
        Node → community ID mapping.

    Returns
    -------
    go.Figure
    """
    stats: dict = {}
    for node in G.nodes():
        cid = node_community.get(node, 0)
        if cid not in stats:
            stats[cid] = {"S": 0, "I": 0, "R": 0}
        s = final_state.get(node, "S")
        stats[cid][s] = stats[cid].get(s, 0) + 1

    def _inf_pct(d):
        total = d["S"] + d["I"] + d.get("R", 0)
        return d["I"] / max(total, 1)

    items = sorted(stats.items(), key=lambda x: _inf_pct(x[1]), reverse=True)[:20]

    labels  = [f"C{cid}" for cid, _ in items]
    totals  = [d["S"] + d["I"] + d.get("R", 0) for _, d in items]
    inf_p   = [d["I"]         / max(t, 1) * 100 for (_, d), t in zip(items, totals)]
    rec_p   = [d.get("R", 0) / max(t, 1) * 100 for (_, d), t in zip(items, totals)]
    sus_p   = [d["S"]         / max(t, 1) * 100 for (_, d), t in zip(items, totals)]
    sizes   = [f"n={t}" for t in totals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Infected %", x=labels, y=inf_p,
        marker_color="#ef4444",
        hovertemplate="Community %{x}<br>Infected: %{y:.1f}%<br>%{customdata}<extra></extra>",
        customdata=sizes,
    ))
    fig.add_trace(go.Bar(
        name="Recovered %", x=labels, y=rec_p,
        marker_color="#3b82f6",
        hovertemplate="Community %{x}<br>Recovered: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Susceptible %", x=labels, y=sus_p,
        marker_color="#64748b",
        hovertemplate="Community %{x}<br>Susceptible: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="🏘️ Infection Spread Across Communities (Top 20 by infection %)",
            font=dict(size=15, color="#f1f5f9"),
        ),
        xaxis=dict(title="Community", gridcolor="#1e293b", linecolor="#334155"),
        yaxis=dict(title="Nodes (%)", range=[0, 100],
                   gridcolor="#1e293b", linecolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.6)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
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

        # ── R₀ Gauge + Herd Immunity ─────────────────────────────────────
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='color:#94a3b8; font-size:0.95rem; margin-bottom:0;'>"
            "📐 Epidemiological Metrics</h4>",
            unsafe_allow_html=True,
        )
        _r0, _hit = compute_r0_hit(beta, gamma, G)
        st.plotly_chart(plot_r0_gauge(_r0), use_container_width=True)

        _budget_pct = (min(budget, G.number_of_nodes()) / G.number_of_nodes()) * 100
        _above      = _budget_pct >= _hit

        st.metric(
            "R₀ — Basic Reproduction Number", f"{_r0:.2f}",
            delta="Epidemic will spread" if _r0 > 1 else "Epidemic dies out",
            delta_color="inverse" if _r0 > 1 else "normal",
            help="R₀ = β × avg_degree / γ  (mean-field approximation)",
        )
        st.metric(
            "Herd Immunity Threshold", f"{_hit:.1f}%",
            help="Fraction of population to immunize to stop exponential spread.",
        )
        st.metric(
            "Your Budget Coverage", f"{_budget_pct:.1f}%",
            delta=f"{'✅ Exceeds' if _above else '⚠️ Below'} HIT ({_hit:.1f}%)",
            delta_color="normal" if _above else "inverse",
        )

    # ── TEMPORAL ANIMATION (full width) ──────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#1e293b; margin:1.5rem 0;'/>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='background:linear-gradient(90deg,#38bdf8,#818cf8);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-weight:700;'>⏱ Temporal Spread Animation</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#64748b;'>Press <b style='color:#38bdf8;'>▶ Play</b> "
        "to watch the epidemic wave evolve step by step. "
        "Drag the slider to jump to any time step.</p>",
        unsafe_allow_html=True,
    )

    anim_col1, anim_col2 = st.columns(2, gap="large")
    with anim_col1:
        st.markdown(
            "<p style='color:#ef4444; font-weight:600; margin-bottom:0;'>"
            "🔴 Baseline — No Immunization</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_animated_spread(history_baseline, model_type, "Baseline"),
            use_container_width=True,
        )
    with anim_col2:
        st.markdown(
            f"<p style='color:#22c55e; font-weight:600; margin-bottom:0;'>"
            f"🟢 Immunized — {strategy.capitalize()} Strategy</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_animated_spread(history_immunized, model_type, "Immunized"),
            use_container_width=True,
        )

    # ── COMMUNITY DETECTION (full width) ─────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#1e293b; margin:1.5rem 0;'/>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='background:linear-gradient(90deg,#f472b6,#818cf8);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-weight:700;'>🏘️ Community-Level Spread Analysis</h2>",
        unsafe_allow_html=True,
    )

    with st.spinner("Detecting communities via Greedy Modularity Optimisation …"):
        _node_comm, _n_comm = detect_communities(G)

    _comm_col1, _comm_col2, _comm_col3 = st.columns(3)
    _comm_col1.metric("Communities Detected", f"{_n_comm:,}")

    # Most-infected community
    _comm_infected = {}
    for node in G.nodes():
        cid = _node_comm.get(node, 0)
        _comm_infected[cid] = _comm_infected.get(cid, 0) + (1 if final_state_b.get(node) == "I" else 0)
    _worst_comm    = max(_comm_infected, key=_comm_infected.get) if _comm_infected else 0
    _worst_count   = _comm_infected.get(_worst_comm, 0)
    _comm_col2.metric("Most-Infected Community", f"C{_worst_comm}", delta=f"{_worst_count} infected nodes")
    _comm_col3.metric("Avg Community Size", f"{G.number_of_nodes() // max(_n_comm, 1):,} nodes")

    st.markdown(
        f"<small style='color:#64748b;'>Detected <b style='color:#38bdf8;'>{_n_comm}</b> "
        "communities using Greedy Modularity. Showing top 20 by infection %. "
        "Hover over bars to see community size.</small>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_community_heatmap(G, final_state_b, _node_comm),
        use_container_width=True,
    )

    # ── NLP MISINFORMATION CLASSIFICATION (full width) ───────────────────────
    st.markdown(
        "<hr style='border-color:#1e293b; margin:1.5rem 0;'/>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='background:linear-gradient(90deg,#f59e0b,#ef4444);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-weight:700;'>🤖 NLP Misinformation Classification</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#64748b;'>Each node is assigned synthetic text content and "
        "classified as <b style='color:#ef4444;'>FAKE</b> or <b style='color:#22c55e;'>REAL</b> "
        "using a <b>TF-IDF + Logistic Regression</b> pipeline trained on a labelled corpus.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("🤖 Training NLP classifier and classifying nodes …"):
        _nlp_pipeline  = train_detector(verbose=False)
        _node_texts    = generate_node_texts(G, fake_fraction=0.35)
        _node_labels   = classify_nodes(G, _nlp_pipeline, _node_texts)
        _det_summary   = detection_summary(_node_labels)

    # ── Top metrics row ───────────────────────────────────────────────────
    _nlp_m1, _nlp_m2, _nlp_m3, _nlp_m4 = st.columns(4)
    _nlp_m1.metric("Total Nodes Classified", f"{_det_summary['total']:,}")
    _nlp_m2.metric(
        "🔴 FAKE Nodes",
        f"{_det_summary['fake']:,}",
        delta=f"{_det_summary['fake_pct']:.1f}% of network",
        delta_color="inverse",
    )
    _nlp_m3.metric(
        "🟢 REAL Nodes",
        f"{_det_summary['real']:,}",
        delta=f"{_det_summary['real_pct']:.1f}% of network",
        delta_color="normal",
    )
    # Fake influencers — nodes in top-20 by degree labelled FAKE
    _top_degree_nodes = centrality_df.sort_values("degree", ascending=False).head(20).index.tolist()
    _fake_influencers = [n for n in _top_degree_nodes if _node_labels.get(n) == "FAKE"]
    _nlp_m4.metric(
        "⚠️ Fake Influencers (Top-20)",
        f"{len(_fake_influencers)}",
        delta=f"out of top-20 by degree",
        delta_color="inverse" if _fake_influencers else "normal",
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Donut chart + Node table side by side ────────────────────────────
    _nlp_chart_col, _nlp_table_col = st.columns([4, 6], gap="large")

    with _nlp_chart_col:
        _donut = go.Figure(go.Pie(
            labels=["FAKE", "REAL"],
            values=[_det_summary["fake"], _det_summary["real"]],
            hole=0.62,
            marker=dict(
                colors=["#ef4444", "#22c55e"],
                line=dict(color="#0d1117", width=3),
            ),
            textfont=dict(color="#e2e8f0", size=13),
            hovertemplate="%{label}: %{value:,} nodes (%{percent})<extra></extra>",
        ))
        _donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1,
                        font=dict(size=13)),
            margin=dict(l=10, r=10, t=40, b=10),
            height=300,
            annotations=[dict(
                text=f"<b>{_det_summary['fake_pct']:.1f}%</b><br>FAKE",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color="#ef4444"),
            )],
            title=dict(text="Node Classification Split", font=dict(size=14, color="#f1f5f9")),
        )
        st.plotly_chart(_donut, use_container_width=True)

    with _nlp_table_col:
        st.markdown("##### 📋 Classified Node Sample")
        _label_filter = st.radio(
            "Filter by label",
            options=["ALL", "FAKE", "REAL"],
            horizontal=True,
            key="nlp_label_filter",
        )
        # Build display dataframe
        _rows = []
        for _node, _lbl in list(_node_labels.items())[:500]:  # cap at 500 for display
            if _label_filter == "ALL" or _lbl == _label_filter:
                _rows.append({
                    "Node ID": _node,
                    "Label": _lbl,
                    "Content": _node_texts.get(_node, ""),
                })
        _node_df = pd.DataFrame(_rows[:150])  # show max 150 rows in table
        if not _node_df.empty:
            def _colour_label(v):
                if v == "FAKE":
                    return "background-color:#3d1515; color:#ef4444; font-weight:600"
                return "background-color:#0f2d1f; color:#22c55e; font-weight:600"
            st.dataframe(
                _node_df.style.applymap(_colour_label, subset=["Label"]),
                use_container_width=True,
                height=300,
            )
        else:
            st.info("No nodes match the selected filter.")

    # ── Live interactive classifier ───────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='color:#f1f5f9;'>🔬 Live Text Classifier</h4>"
        "<p style='color:#64748b; margin-top:-0.5rem;'>Type any headline or claim below "
        "and get an instant FAKE / REAL prediction with confidence score.</p>",
        unsafe_allow_html=True,
    )
    _live_col1, _live_col2 = st.columns([7, 3], gap="large")
    with _live_col1:
        _user_text = st.text_area(
            "Enter text to classify",
            placeholder="e.g. Scientists confirm 5G towers cause COVID-19 spread …",
            height=100,
            key="nlp_live_input",
            label_visibility="collapsed",
        )
    with _live_col2:
        st.markdown("<br/>", unsafe_allow_html=True)
        _classify_btn = st.button("⚡ Classify", type="primary", use_container_width=True, key="nlp_classify_btn")

    if _classify_btn and _user_text.strip():
        import numpy as np
        _proba = _nlp_pipeline.predict_proba([_user_text.strip()])[0]  # [P(REAL), P(FAKE)]
        _pred_idx = int(_nlp_pipeline.predict([_user_text.strip()])[0])
        _pred_label = "FAKE" if _pred_idx == 1 else "REAL"
        _confidence = _proba[_pred_idx] * 100
        _colour = "#ef4444" if _pred_label == "FAKE" else "#22c55e"
        _icon   = "🔴" if _pred_label == "FAKE" else "🟢"
        st.markdown(
            f"""
            <div style='padding:1.2rem 1.5rem; border-radius:12px;
                        border:1px solid {_colour}40;
                        background:linear-gradient(135deg,{_colour}15,{_colour}05);
                        margin-top:0.5rem;'>
                <span style='font-size:1.8rem;'>{_icon}</span>
                <span style='font-size:1.5rem; font-weight:700; color:{_colour};
                             margin-left:0.5rem;'>{_pred_label}</span>
                <span style='color:#94a3b8; margin-left:1rem;'>Confidence: 
                    <b style='color:{_colour};'>{_confidence:.1f}%</b></span>
                <p style='color:#94a3b8; margin:0.6rem 0 0; font-size:0.9rem;'>
                    P(REAL) = {_proba[0]*100:.1f}% &nbsp;|&nbsp; P(FAKE) = {_proba[1]*100:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif _classify_btn:
        st.warning("Please enter some text to classify.")

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
