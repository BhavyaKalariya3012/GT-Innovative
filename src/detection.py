"""
detection.py
------------
Misinformation detection module using TF-IDF + Logistic Regression.

Provides:
  - A built-in synthetic labelled corpus (no external data required)
  - train_detector()      : trains and returns a sklearn Pipeline
  - classify_nodes()      : labels nodes FAKE / REAL given their text content
  - detection_summary()   : prints and returns a summary dict
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ---------------------------------------------------------------------------
# Built-in Synthetic Corpus
# ---------------------------------------------------------------------------

# Each tuple: (text, label)  label = 1 → FAKE,  label = 0 → REAL
_CORPUS: List[Tuple[str, int]] = [
    # ── FAKE samples ────────────────────────────────────────────────────────
    ("Scientists confirm 5G towers cause COVID-19 spread in cities", 1),
    ("Vaccines contain microchips to track your location secretly", 1),
    ("Government hiding cure for cancer to protect big pharma profits", 1),
    ("Drinking bleach kills coronavirus according to secret document", 1),
    ("Chemtrails are spraying mind-control chemicals on populations", 1),
    ("Flat earth society reveals NASA conspiracy to hide planet shape", 1),
    ("Bill Gates planning global population reduction through vaccines", 1),
    ("Moon landing was filmed in Hollywood studio, new evidence shows", 1),
    ("Eating garlic daily completely prevents all viral infections", 1),
    ("New world order controls every government leader in the world", 1),
    ("Climate change is a hoax invented by globalists to control energy", 1),
    ("Microwave ovens cause cancer according to suppressed research", 1),
    ("Election results rigged by dominion voting machines worldwide", 1),
    ("Alien technology found in area 51 confirmed by whistleblower", 1),
    ("Chemical in tap water causes infertility said secret memo", 1),
    ("Covid vaccines alter human DNA permanently says researcher", 1),
    ("Birds are not real they are government surveillance drones", 1),
    ("This fruit cures diabetes in three days doctors dont want you knowing", 1),
    ("Illuminati confirmed to control Hollywood and media completely", 1),
    ("Sunscreen contains chemicals that cause the skin cancer it claims to prevent", 1),
    ("Ancient pyramids built by aliens not humans new evidence emerges", 1),
    ("Secret military base found on moon confirms nasa lies", 1),
    ("Drinking hydrogen peroxide cures all diseases naturally at home", 1),
    ("Deep state shadow government controls all elections globally", 1),
    ("Antifa paid by billionaire soros to cause chaos in major cities", 1),
    ("WiFi radiation causes brain tumours new whistleblower reveals", 1),
    ("Gold standard removed to control population through debt forever", 1),
    ("Natural remedies cured cancer patient doctors called impossible", 1),
    ("QAnon decoded message reveals truth about global child trafficking ring", 1),
    ("Face masks cause oxygen deprivation leading to brain damage doctors warn", 1),
    ("Corona virus was patented years before outbreak proof found online", 1),
    ("Pfizer executive admits vaccine side effects were hidden from public", 1),
    ("Shadow government plans microchip implant in all citizens by 2025", 1),
    ("Crop circles decoded reveal alien warning about upcoming disaster", 1),
    ("Fluoride in water is mind control chemical added by government agencies", 1),
    ("Major earthquake predicted using secret HAARP weather machine signal", 1),
    ("Rothschild family controls all central banks to enslave humanity", 1),
    ("Government satellites controlling weather patterns to destroy crops", 1),
    ("New evidence shows shakespeare was actually a woman in disguise", 1),
    ("Secret tunnel network discovered under major capital cities worldwide", 1),
    ("Big pharma suppresses natural cancer cure to maintain profits", 1),
    ("World economic forum plans great reset to eliminate private property", 1),
    ("Mandatory vaccines will contain gps tracking nano technology chips", 1),
    ("Secret society controls supreme court judges decisions worldwide", 1),
    ("Underground bunkers built by elite for upcoming pole shift catastrophe", 1),
    ("Water supplies secretly contaminated to lower IQ of population", 1),
    ("Time traveller photographed at 1940s event revealed by historian", 1),
    ("New study shows meditation cures terminal cancer in three weeks", 1),
    ("Hidden codes in dollar bill prove masonic control of united states", 1),
    ("Hollow earth theory confirmed by satellite imaging data leaked online", 1),
    # ── REAL samples ────────────────────────────────────────────────────────
    ("Study published in Nature confirms efficacy of mRNA COVID vaccine", 0),
    ("WHO releases updated guidelines for pandemic preparedness protocols", 0),
    ("US Federal Reserve raises interest rates to combat inflation", 0),
    ("Scientists discover new exoplanet in habitable zone of nearby star", 0),
    ("Climate scientists report record ocean temperatures in 2024", 0),
    ("New cancer drug approved by FDA after phase three clinical trials", 0),
    ("European Parliament passes digital markets act to regulate big tech", 0),
    ("Researchers identify gene mutation linked to early onset Alzheimer disease", 0),
    ("SpaceX successfully launches Starship on third test flight", 0),
    ("Global renewable energy capacity doubled in past decade says IEA", 0),
    ("Oxford University study shows sleep deprivation impairs immune response", 0),
    ("G20 leaders agree on framework for global minimum corporate tax", 0),
    ("Archaeologists uncover ancient Roman settlement during highway excavation", 0),
    ("Meta releases open source large language model for research use", 0),
    ("Antarctic ice sheet data shows acceleration of glacier melting rate", 0),
    ("NASA publishes high resolution images of Mars surface from Perseverance", 0),
    ("Peer reviewed research confirms benefits of Mediterranean diet for heart", 0),
    ("International court rules in favour of indigenous land rights claim", 0),
    ("New submarine cable to boost internet connectivity across Africa", 0),
    ("Economists predict moderate GDP growth across emerging markets in 2025", 0),
    ("University researchers develop biodegradable plastic from seaweed extract", 0),
    ("Geneva convention updated to address autonomous weapons systems", 0),
    ("Child mortality rates reach historic low according to UNICEF report", 0),
    ("Scientists sequence genome of newly discovered deep sea species", 0),
    ("EU approves 1 trillion euro green deal investment package", 0),
    ("Electric vehicle sales surpass petrol cars in Norway for first time", 0),
    ("Long COVID symptoms affect cognitive function study finds", 0),
    ("Record number of women elected to parliament in recent elections", 0),
    ("New clean water initiative brings safe drinking water to rural Africa", 0),
    ("Astronomers detect strongest gravitational wave signal recorded yet", 0),
    ("WHO declares end of mpox public health emergency of international concern", 0),
    ("Stanford researchers create AI model to detect early pancreatic cancer", 0),
    ("Cyclone relief efforts coordinated by ASEAN nations after landfall", 0),
    ("Solar panel efficiency record broken by new perovskite cell design", 0),
    ("US CDC updates masking guidance based on latest transmission data", 0),
    ("CERN detects new particle interaction not predicted by standard model", 0),
    ("International Monetary Fund approves emergency loan for struggling economy", 0),
    ("Genome editing technique shows promise for sickle cell disease patients", 0),
    ("Ocean plastic cleanup project removes 200 tonnes from Pacific gyre", 0),
    ("New high speed rail line opens connecting two major european cities", 0),
    ("Scientists confirm ancient human migration route across Bering land bridge", 0),
    ("Global hunger index shows improvement in food security in south asia", 0),
    ("Drug resistance rising in common bacterial infections says WHO report", 0),
    ("Mapping project reveals hidden medieval city beneath modern settlement", 0),
    ("Clinical trial shows new malaria vaccine reduces child mortality by 30 percent", 0),
    ("Central banks coordinate to prevent contagion from bank failures", 0),
    ("AI system beats human experts at protein folding prediction challenge", 0),
    ("World heritage site status granted to ancient trade route ruins", 0),
    ("Breakthrough in nuclear fusion yields net energy gain for two seconds", 0),
    ("Drought conditions worsen in horn of africa food crisis worsens", 0),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_detector(
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Pipeline:
    """Train a TF-IDF + Logistic Regression misinformation classifier.

    Uses the built-in synthetic corpus (100 samples). No external files needed.

    Parameters
    ----------
    test_size : float
        Fraction of data held out for evaluation. Default 0.2.
    random_state : int
        Random seed for reproducibility. Default 42.
    verbose : bool
        If True, prints training metrics. Default True.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for :func:`classify_nodes`.
    """
    texts  = [t for t, _ in _CORPUS]
    labels = [l for _, l in _CORPUS]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )),
    ])

    pipeline.fit(X_train, y_train)

    if verbose:
        y_pred = pipeline.predict(X_test)
        print("\n" + "=" * 45)
        print("  🤖 Misinformation Detector — Training Report")
        print("=" * 45)
        report = classification_report(
            y_test, y_pred,
            target_names=["REAL", "FAKE"],
            zero_division=0,
        )
        print(report)

    return pipeline


def classify_nodes(
    G: nx.Graph,
    pipeline: Pipeline,
    node_texts: Dict[Any, str],
) -> Dict[Any, str]:
    """Label every node in *G* as 'FAKE' or 'REAL'.

    Nodes not present in *node_texts* are labelled 'UNKNOWN'.

    Parameters
    ----------
    G : nx.Graph
        Social network graph.
    pipeline : Pipeline
        Fitted sklearn pipeline from :func:`train_detector`.
    node_texts : dict
        Mapping ``node_id → text content`` for each node.

    Returns
    -------
    dict
        Mapping ``node_id → 'FAKE' | 'REAL' | 'UNKNOWN'``.
    """
    labels: Dict[Any, str] = {}
    nodes_with_text = [n for n in G.nodes() if n in node_texts]

    if nodes_with_text:
        texts  = [node_texts[n] for n in nodes_with_text]
        preds  = pipeline.predict(texts)
        for node, pred in zip(nodes_with_text, preds):
            labels[node] = "FAKE" if pred == 1 else "REAL"

    for node in G.nodes():
        if node not in labels:
            labels[node] = "UNKNOWN"

    return labels


def generate_node_texts(
    G: nx.Graph,
    fake_fraction: float = 0.35,
    random_state: int = 42,
) -> Dict[Any, str]:
    """Assign synthetic text to every node for demo/testing purposes.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    fake_fraction : float
        Fraction of nodes assigned fake content. Default 0.35.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Mapping ``node_id → text``.
    """
    rng = random.Random(random_state)
    fake_texts = [t for t, l in _CORPUS if l == 1]
    real_texts = [t for t, l in _CORPUS if l == 0]
    node_texts: Dict[Any, str] = {}
    for node in G.nodes():
        if rng.random() < fake_fraction:
            node_texts[node] = rng.choice(fake_texts)
        else:
            node_texts[node] = rng.choice(real_texts)
    return node_texts


def detection_summary(
    labels: Dict[Any, str],
    node_texts: Dict[Any, str] | None = None,
) -> Dict[str, Any]:
    """Print and return a summary of detection results.

    Parameters
    ----------
    labels : dict
        Output of :func:`classify_nodes`.
    node_texts : dict, optional
        If provided, prints example fake node texts for review.

    Returns
    -------
    dict
        ``{'total': int, 'fake': int, 'real': int, 'unknown': int,
           'fake_pct': float, 'real_pct': float}``
    """
    total   = len(labels)
    fake    = sum(1 for v in labels.values() if v == "FAKE")
    real    = sum(1 for v in labels.values() if v == "REAL")
    unknown = sum(1 for v in labels.values() if v == "UNKNOWN")

    fake_pct = fake / max(total, 1) * 100
    real_pct = real / max(total, 1) * 100

    print("\n" + "=" * 45)
    print("  📊 Detection Results Summary")
    print("=" * 45)
    print(f"  Total nodes classified : {total:,}")
    print(f"  🔴 FAKE nodes          : {fake:,}  ({fake_pct:.1f}%)")
    print(f"  🟢 REAL nodes          : {real:,}  ({real_pct:.1f}%)")
    if unknown:
        print(f"  ⚪ UNKNOWN nodes       : {unknown:,}")
    print("=" * 45 + "\n")

    return {
        "total":    total,
        "fake":     fake,
        "real":     real,
        "unknown":  unknown,
        "fake_pct": round(fake_pct, 2),
        "real_pct": round(real_pct, 2),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.graph_builder import load_graph

    G = load_graph()
    pipeline   = train_detector(verbose=True)
    node_texts = generate_node_texts(G)
    labels     = classify_nodes(G, pipeline, node_texts)
    detection_summary(labels, node_texts)
