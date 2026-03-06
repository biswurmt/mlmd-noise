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

_KG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge-graphs")
)
PKL_PATH = os.path.join(_KG_DIR, "triage_knowledge_graph.pkl")


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


def load_graph_json() -> dict:
    """Return {"nodes": [...], "edges": [...]} with all attributes
    serialised to JSON-safe Python types.

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
