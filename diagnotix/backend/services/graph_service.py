"""graph_service.py
===================
Loads the serialised NetworkX DiGraph and converts it to a JSON-serialisable
dict so FastAPI can return it to the frontend visualiser.

All pandas/numpy non-serialisable values (NaN, numpy integers/floats, numpy
arrays) are normalised to plain Python types or None before returning.

Edge attribute naming note
--------------------------
NetworkX edges carry a "source" attribute that stores the GUIDELINE name
(e.g. "AHA/ACC").  That key collides with the "source" key that
react-force-graph uses to identify the source *node*.  We rename the
edge attribute to "guideline_source" so the two meanings never overlap.
"""

import math
import os
import pickle
from typing import Any

from dotenv import load_dotenv

load_dotenv(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")),
    override=False,
)

_KG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge-graphs")
)
_pkl_file = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
PKL_PATH = os.path.join(_KG_DIR, _pkl_file)


def _clean(v: Any) -> Any:
    """Recursively convert non-JSON-safe values to safe equivalents."""
    if v is None:
        return None

    if isinstance(v, float) and math.isnan(v):
        return None

    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]

    try:
        import numpy as np  # noqa: F401
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            val = float(v)
            return None if math.isnan(val) else val
        if isinstance(v, np.ndarray):
            return [_clean(x) for x in v.tolist()]
        if isinstance(v, np.bool_):
            return bool(v)
    except ImportError:
        pass

    return v


def load_test_nodes() -> list[dict]:
    """Return a sorted list of {"id", "label"} dicts for every Diagnostic_Test
    node in the graph.  Returns an empty list if the PKL does not exist yet.
    """
    if not os.path.exists(PKL_PATH):
        return []
    with open(PKL_PATH, "rb") as f:
        G = pickle.load(f)
    tests = [
        {
            "id": n,
            "label": n.split(": ", 1)[1] if ": " in n else n,
        }
        for n, d in G.nodes(data=True)
        if d.get("type") == "Diagnostic_Test"
    ]
    return sorted(tests, key=lambda x: x["label"])


def load_graph_json(pathway: str | None = None) -> dict:
    """Return {"nodes": [...], "edges": [...]} with all attributes
    serialised to JSON-safe Python types.

    If *pathway* is provided (and is not "All Pathways"), the graph is
    filtered to the 1-hop subgraph centred on that Diagnostic_Test node —
    the same cluster the frontend previously computed client-side.

    Node objects include a "node_type" alias for the "type" field so
    the frontend can use a consistent key regardless of PKL version.

    Edge objects rename the "source" attribute (guideline name) to
    "guideline_source" to avoid collision with the react-force-graph
    "source" field (source node ID).
    """
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(
            f"Knowledge graph not found at '{PKL_PATH}'. "
            "Run 'python knowledge-graphs/build_kg.py' first."
        )

    with open(PKL_PATH, "rb") as f:
        G = pickle.load(f)

    if pathway and pathway.lower() != "all pathways":
        # Resolve the test node ID: accept "ECG" or "Test: ECG"
        test_node = f"Test: {pathway}"
        if test_node not in G:
            if pathway in G and G.nodes[pathway].get("type") == "Diagnostic_Test":
                test_node = pathway
            else:
                raise ValueError(f"Pathway '{pathway}' not found in graph.")

        # Build the 1-hop cluster (matches the previous client-side logic)
        cluster = {test_node}
        for u, _ in G.in_edges(test_node):
            cluster.add(u)
        for _, v in G.out_edges(test_node):
            cluster.add(v)
        G = G.subgraph(cluster)

    nodes = []
    for node_id, attrs in G.nodes(data=True):
        node: dict = {"id": node_id}
        for k, v in attrs.items():
            node[k] = _clean(v)
        # Expose both "type" (raw) and "node_type" (frontend-friendly alias)
        if "type" in attrs:
            node["node_type"] = attrs["type"]
        nodes.append(node)

    edges = []
    for src, tgt, attrs in G.edges(data=True):
        edge: dict = {"source": src, "target": tgt}
        for k, v in attrs.items():
            # Rename the guideline "source" attr so it never overwrites
            # the graph-topology "source" (= src node ID) set above.
            key = "guideline_source" if k == "source" else k
            edge[key] = _clean(v)
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def get_existing_node_ids() -> set[str]:
    """Return the set of node IDs currently in the serialised graph.
    Returns an empty set if the PKL does not exist yet.
    """
    if not os.path.exists(PKL_PATH):
        return set()
    with open(PKL_PATH, "rb") as f:
        G = pickle.load(f)
    return set(G.nodes())
