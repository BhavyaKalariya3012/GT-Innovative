# Misinformation Propagation & Immunization Simulator

A Streamlit-powered dashboard that models how misinformation spreads on social networks using SIR/SIS epidemic models and lets you evaluate six immunization strategies.

---

## Features

| Feature | Details |
|---|---|
| **Graph Source** | SNAP Twitter ego-network or synthetic Barabási-Albert fallback |
| **Epidemic Models** | SIR and SIS (pure-Python, no NDlib) |
| **Centrality Metrics** | Degree, Betweenness, Eigenvector, PageRank, Closeness, K-Core |
| **Immunization Strategies** | Random, Degree, Betweenness, PageRank, Acquaintance, K-Core |
| **Visualisations** | Plotly epidemic curves, strategy comparison bar chart, Pyvis network |

---

## Project Structure

```
misinfo-propagation/
├── data/                     # Place twitter_combined.txt here
├── src/
│   ├── __init__.py
│   ├── graph_builder.py      # Graph loading & preprocessing
│   ├── centrality.py         # Six centrality measures + composite score
│   ├── epidemic_model.py     # SIR and SIS models (pure Python)
│   └── immunization.py       # Six immunization strategies + benchmark
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create & activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Download the SNAP Twitter Dataset

For a real-world social network:

1. Visit **https://snap.stanford.edu/data/ego-Twitter.html**
2. Download `twitter_combined.txt.gz`
3. Extract the file:
   ```bash
   gunzip twitter_combined.txt.gz
   ```
4. Move `twitter_combined.txt` into the `data/` folder:
   ```
   misinfo-propagation/data/twitter_combined.txt
   ```

> **Note:** If the file is not found the app automatically generates a synthetic Barabási-Albert graph (1 000 nodes) so the dashboard works without any dataset.

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage

1. Use the **sidebar** to choose:
   - Epidemic model (SIR / SIS)
   - Infection probability β and recovery probability γ
   - Immunization strategy and node budget
2. Click **Run Simulation**.
3. Examine:
   - The **epidemic curve** comparing baseline vs immunized spread
   - The **strategy comparison bar chart** with error bars
   - The **interactive network** coloured by node state
   - The **metric cards** showing peak infection and % spread reduction

---

## Module Overview

### `src/graph_builder.py`
Loads `twitter_combined.txt` (or synthetic BA graph) → extracts the giant weakly-connected component → removes self-loops → prints graph statistics.

### `src/centrality.py`
Computes degree, betweenness (k=500 approximation), eigenvector, PageRank, closeness, and k-core. Returns a DataFrame sorted by a composite score (mean of normalised degree, betweenness, eigenvector, and PageRank).

### `src/epidemic_model.py`
Pure-Python Monte-Carlo SIR and SIS simulations. Returns per-step history dicts and final node states.

### `src/immunization.py`
Six strategies targeting different structural properties. `benchmark_strategies()` runs 10 Monte-Carlo SIR replications per strategy and returns mean/std peak infection counts.

---

## Requirements

```
networkx>=3.0
streamlit>=1.28.0
plotly>=5.15.0
pyvis>=0.3.2
pandas>=2.0.0
scipy>=1.11.0
numpy>=1.24.0
```

---

## License

MIT
