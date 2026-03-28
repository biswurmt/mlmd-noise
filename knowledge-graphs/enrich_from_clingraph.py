"""
enrich_from_clingraph.py
========================
Enriches the triage KG with ClinGraph neighbourhood data.

ClinGraph (Harvard Zitnik Lab / Harvard Dataverse doi:10.7910/DVN/Z6H1A8) is a
153K-node, ~3M-edge clinical knowledge graph spanning SNOMED-CT, ICD-10-CM,
LOINC, RxNorm, ATC, CPT, and PheCode.  Our existing triage KG nodes already
carry SNOMED CA, LOINC, ICD-10, and RxCUI codes — those serve as a direct
bridge to find clinically relevant neighbours in ClinGraph.

ClinGraph file format (actual schema):
    ClinGraph_nodes.csv  — tab-separated: node_id, node_name, ntype, node_index
        node_id format: "{code}:{vocab_lowercase}"  e.g. "29857009:snomedct_us"
    ClinGraph_edges.csv  — tab-separated (denormalized):
        index, node_id_x, node_name_x, ntype_x,
        relationship,
        node_id_y, node_name_y, ntype_y,
        node_index_x, node_index_y, edge_index

Reads:
    triage_knowledge_graph.pkl             (base guideline KG)
    clingraph_data/ClinGraph_nodes.csv
    clingraph_data/ClinGraph_edges.csv

Writes:
    triage_knowledge_graph_enriched.pkl    (enriched KG — base graph intact)

Usage:
    python enrich_from_clingraph.py
    python enrich_from_clingraph.py --base-pkl path/to/base.pkl
    python enrich_from_clingraph.py --max-new-nodes 300
"""

import argparse
import copy
import os
import pickle
import sys

import networkx as nx
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary constants (exact ntype values used in ClinGraph_nodes.csv)
# ─────────────────────────────────────────────────────────────────────────────

# Vocabularies to keep; drop ICD9CM, CPT, PHECODE, UMLS_CUI
_KEEP_VOCABS = {"SNOMEDCT_US", "ICD10CM", "LNC", "RXNORM", "ATC"}

# Vocab → triage KG node type (SNOMED is refined by name heuristic below)
_VOCAB_TO_NODE_TYPE: dict[str, str] = {
    "SNOMEDCT_US": "Symptom",       # refined by _infer_snomed_type()
    "ICD10CM":     "Condition",
    "LNC":         "Diagnostic_Test",
    "RXNORM":      "Treatment",
    "ATC":         "Treatment",
}

# Node type → display prefix (mirrors _NODE_PREFIX in build_kg.py)
_NODE_TYPE_TO_PREFIX: dict[str, str] = {
    "Symptom":             "Symptom",
    "Condition":           "Condition",
    "Diagnostic_Test":     "Test",
    "Treatment":           "Treatment",
    "Vital_Sign_Threshold":"Vital",
    "Demographic_Factor":  "Demographic",
    "Risk_Factor":         "Risk Factor",
    "Mechanism_of_Injury": "MOI",
    "Clinical_Attribute":  "Attribute",
}

# KG node attribute → ClinGraph vocab ntype  (for code matching)
_CODE_ATTR_VOCAB: list[tuple[str, str]] = [
    ("snomed_ca_code", "SNOMEDCT_US"),
    ("loinc_code",     "LNC"),
    ("icd10_code",     "ICD10CM"),
    ("rxcui",          "RXNORM"),
]

# SNOMED name substrings that indicate a Condition rather than a Symptom
_CONDITION_NAME_HINTS = (
    "disease", "disorder", "syndrome", "failure", "infarction",
    "carcinoma", "tumour", "tumor", "infection", "insufficiency",
    "neoplasm", "torsion", "fracture", "appendicitis", "occlusion",
    "aneurysm", "thrombosis", "embolism", "cardiomyopathy",
)

# ICD-10-CM code prefixes that classify as Symptom rather than Condition.
# R-codes are "Symptoms, signs and abnormal clinical and laboratory findings"
# (ICD-10-CM Chapter 18) — these are findings, not diagnoses.
# Certain M-codes (M25, M54, M79) are pain/symptom codes rather than diseases.
_ICD10CM_SYMPTOM_PREFIXES = ("R",)
_ICD10CM_SYMPTOM_SUBPREFIXES = ("M25.", "M54.", "M79.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_node_id(node_id: str) -> tuple[str, str]:
    """
    Split a ClinGraph node_id of the form "{code}:{vocab_lower}" into
    (code, VOCAB_UPPER).  Examples:
        "29857009:snomedct_us" → ("29857009", "SNOMEDCT_US")
        "R10.0:icd10cm"        → ("R10.0",    "ICD10CM")
        "718-7:lnc"            → ("718-7",    "LNC")
    """
    parts = str(node_id).rsplit(":", 1)
    if len(parts) == 2:
        return parts[0], parts[1].upper()
    return str(node_id), ""


def _infer_snomed_type(name: str) -> str:
    """Heuristic: classify a SNOMED concept as Condition or Symptom from its name."""
    lower = name.lower()
    if any(hint in lower for hint in _CONDITION_NAME_HINTS):
        return "Condition"
    return "Symptom"


def _vocab_to_node_type(ntype: str, name: str, code: str = "") -> str:
    if ntype == "SNOMEDCT_US":
        return _infer_snomed_type(name)
    if ntype == "ICD10CM" and code:
        # R-codes = signs/symptoms/abnormal findings → Symptom
        if code.startswith(_ICD10CM_SYMPTOM_PREFIXES):
            return "Symptom"
        # Specific M-codes that are pain/symptom codes → Symptom
        if code.startswith(_ICD10CM_SYMPTOM_SUBPREFIXES):
            return "Symptom"
        # All other ICD-10-CM codes → Condition (I, K, N, O, S, etc.)
        return "Condition"
    return _VOCAB_TO_NODE_TYPE.get(ntype, "Symptom")


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_clingraph_nodes(node_csv: str) -> pd.DataFrame:
    """
    Load ClinGraph_nodes.csv.
    Returns DataFrame with columns: node_id, node_name, ntype, code.
    """
    print(f"  Reading {node_csv} ...")
    df = pd.read_csv(node_csv, sep=None, engine="python")  # auto-detect tab vs comma
    print(f"  Columns detected: {list(df.columns)}")

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    required = {"node_id", "node_name", "ntype"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(
            f"ClinGraph_nodes.csv is missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    df = df[["node_id", "node_name", "ntype"]].copy()
    df["node_id"]   = df["node_id"].astype(str).str.strip()
    df["node_name"] = df["node_name"].astype(str).str.strip()
    df["ntype"]     = df["ntype"].astype(str).str.strip().str.upper()

    # Extract plain code from node_id for index building
    df["code"] = df["node_id"].apply(lambda x: _parse_node_id(x)[0])

    print(f"  Loaded {len(df):,} nodes.")
    return df


def load_clingraph_edges(edge_csv: str) -> pd.DataFrame:
    """
    Load ClinGraph_edges.csv (denormalized).
    Returns DataFrame with columns: src, tgt, src_name, tgt_name, src_ntype, tgt_ntype, relation.
    """
    print(f"  Reading {edge_csv} ...")
    df = pd.read_csv(edge_csv, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    required = {"node_id_x", "node_id_y", "relationship", "ntype_x", "ntype_y",
                "node_name_x", "node_name_y"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(
            f"ClinGraph_edges.csv is missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    out = df[["node_id_x", "node_name_x", "ntype_x",
              "relationship",
              "node_id_y", "node_name_y", "ntype_y"]].copy()
    out.columns = ["src", "src_name", "src_ntype",
                   "relation",
                   "tgt", "tgt_name", "tgt_ntype"]

    out["src"]      = out["src"].astype(str).str.strip()
    out["tgt"]      = out["tgt"].astype(str).str.strip()
    out["src_ntype"]= out["src_ntype"].astype(str).str.strip().str.upper()
    out["tgt_ntype"]= out["tgt_ntype"].astype(str).str.strip().str.upper()

    print(f"  Loaded {len(out):,} edges.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Code Index
# ─────────────────────────────────────────────────────────────────────────────

def build_code_index(cg_nodes: pd.DataFrame) -> dict:
    """
    Build lookup: (NTYPE, code_str) → node_id.
    Only indexes vocabularies we plan to bridge.
    """
    print("  Building vocabulary code → node_id index ...")
    relevant = cg_nodes[cg_nodes["ntype"].isin(_KEEP_VOCABS)]
    index = {
        (row["ntype"], row["code"]): row["node_id"]
        for _, row in relevant.iterrows()
    }
    print(f"  Index contains {len(index):,} mappings.")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# KG → ClinGraph Node Matching
# ─────────────────────────────────────────────────────────────────────────────

def match_kg_to_clingraph(G: nx.DiGraph, code_index: dict) -> dict:
    """
    For each KG node, find matching ClinGraph node_ids via stored ontology codes.
    Returns: {kg_node_id: [cg_node_id, ...]}
    """
    print("  Matching KG nodes to ClinGraph via ontology codes ...")
    mapping: dict[str, list[str]] = {}

    for kg_node, attrs in G.nodes(data=True):
        matches = []
        for attr_key, vocab in _CODE_ATTR_VOCAB:
            raw = attrs.get(attr_key)
            if raw is None:
                continue
            if isinstance(raw, float) and pd.isna(raw):
                continue
            code_str = str(raw).strip()
            if not code_str or code_str.lower() in ("nan", "none", ""):
                continue

            # Try exact code, then integer-stripped version (e.g. "1234.0" → "1234")
            cg_id = code_index.get((vocab, code_str))
            if cg_id is None and "." in code_str:
                cg_id = code_index.get((vocab, code_str.split(".")[0]))
            if cg_id is not None:
                matches.append(cg_id)

        if matches:
            mapping[kg_node] = matches

    matched = len(mapping)
    total   = G.number_of_nodes()
    pct     = 100 * matched // total if total else 0
    print(f"  Matched {matched} / {total} KG nodes ({pct}%).")
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Attribute Enrichment — add cross-vocabulary codes to existing KG nodes
# ─────────────────────────────────────────────────────────────────────────────

def enrich_node_attributes(
    G: nx.DiGraph,
    kg_to_cg: dict,
    neighbourhood: pd.DataFrame,
) -> int:
    """
    For each existing KG node that matched a ClinGraph node, traverse DEFINED
    edges in the neighbourhood to pull cross-vocabulary codes into node attributes.

    Enrichments added (only when the attribute is currently missing/None):
      snomed_ca_code matched → icd10_code    via SNOMEDCT_US → ICD10CM  DEFINED
      icd10_code matched     → snomed_ca_code via ICD10CM → SNOMEDCT_US DEFINED
      rxcui matched          → atc_code       via RXNORM → ATC          DEFINED

    Modifies G in-place. Returns the number of attributes added.
    """
    defined = neighbourhood[neighbourhood["relation"] == "DEFINED"]

    # Build quick lookup: cg_node_id → set of (neighbour_id, ntype) via DEFINED
    defined_map: dict[str, list[tuple]] = collections.defaultdict(list)
    for _, row in defined.iterrows():
        defined_map[row["src"]].append((row["tgt"], row["tgt_ntype"]))
        defined_map[row["tgt"]].append((row["src"], row["src_ntype"]))

    # Build cg_node_id → code lookup from node_id format
    def cg_code(node_id: str) -> str:
        return _parse_node_id(node_id)[0]

    added = 0
    for kg_node, cg_ids in kg_to_cg.items():
        attrs = G.nodes[kg_node]
        for cg_id in cg_ids:
            for nb_id, nb_ntype in defined_map.get(cg_id, []):
                code = cg_code(nb_id)
                if not code:
                    continue
                # SNOMED matched → fill missing icd10cm_code (US Clinical Modification,
                # kept separate from icd10_code (WHO) and icd10ca_code (Canadian))
                if nb_ntype == "ICD10CM" and not attrs.get("icd10cm_code"):
                    G.nodes[kg_node]["icd10cm_code"] = code
                    added += 1
                # ICD10 matched → fill missing snomed_ca_code
                elif nb_ntype == "SNOMEDCT_US" and not attrs.get("snomed_ca_code"):
                    G.nodes[kg_node]["snomed_ca_code"] = code
                    added += 1
                # RxNorm matched → fill missing atc_code
                elif nb_ntype == "ATC" and not attrs.get("atc_code"):
                    G.nodes[kg_node]["atc_code"] = code
                    added += 1

    print(f"  Added {added} cross-vocabulary code attributes to existing nodes.")
    return added


import collections


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_neighbourhood(cg_edges: pd.DataFrame, seed_ids: set) -> pd.DataFrame:
    """Return all ClinGraph edges where at least one endpoint is in seed_ids."""
    mask = cg_edges["src"].isin(seed_ids) | cg_edges["tgt"].isin(seed_ids)
    neighbourhood = cg_edges[mask].copy()
    print(f"  1-hop neighbourhood: {len(neighbourhood):,} edges around {len(seed_ids)} seeds.")
    return neighbourhood


# ─────────────────────────────────────────────────────────────────────────────
# Merge into KG
# ─────────────────────────────────────────────────────────────────────────────

def merge_clingraph_neighbourhood(
    G: nx.DiGraph,
    kg_to_cg: dict,
    neighbourhood: pd.DataFrame,
    max_new_nodes: int = 500,
) -> nx.DiGraph:
    """
    Add ClinGraph neighbour nodes and CLINGRAPH_RELATED edges to a deep copy of G.

    Since ClinGraph_edges.csv is denormalized (name and ntype already present on
    each edge row), we don't need a separate node lookup — all neighbour metadata
    comes directly from the edge row.

    DiGraph note: duplicate (src, tgt) pairs overwrite the earlier edge.  Rows are
    processed in the order they appear in *neighbourhood* (no sorting by weight
    since ClinGraph has no weight column).  The last relationship written for a
    given pair wins; this is consistent with the existing build_kg.py behaviour.
    """
    G_enriched = copy.deepcopy(G)

    all_seed_ids: set = {cg_id for cg_ids in kg_to_cg.values() for cg_id in cg_ids}

    # Reverse map: cg_node_id → kg_node_id (anchor in our KG)
    cg_to_kg: dict[str, str] = {}
    for kg_node, cg_ids in kg_to_cg.items():
        for cg_id in cg_ids:
            cg_to_kg[cg_id] = kg_node

    # Normalised label → existing KG node_id (for dedup without prefix)
    existing_labels: dict[str, str] = {
        nid.split(": ", 1)[-1].lower(): nid
        for nid in G_enriched.nodes()
    }

    stats = {"new_nodes": 0, "new_edges": 0, "dedup_edges": 0, "skipped_vocab": 0}

    for _, row in neighbourhood.iterrows():
        if stats["new_nodes"] >= max_new_nodes:
            break

        src_id = row["src"]
        tgt_id = row["tgt"]
        relation = str(row.get("relation", "related"))

        # Identify which end is the seed (our KG anchor) and which is the neighbour.
        # anchor_is_src tracks original ClinGraph direction so we can preserve it:
        #   anchor was src → edge goes anchor → new_node  (same direction)
        #   anchor was tgt → edge goes new_node → anchor  (e.g. Symptom → Condition)
        if src_id in all_seed_ids and tgt_id not in all_seed_ids:
            anchor_kg    = cg_to_kg.get(src_id)
            nb_id        = tgt_id
            nb_name      = str(row.get("tgt_name", "")).strip()
            nb_ntype     = str(row.get("tgt_ntype", "")).strip().upper()
            anchor_is_src = True
        elif tgt_id in all_seed_ids and src_id not in all_seed_ids:
            anchor_kg    = cg_to_kg.get(tgt_id)
            nb_id        = src_id
            nb_name      = str(row.get("src_name", "")).strip()
            nb_ntype     = str(row.get("src_ntype", "")).strip().upper()
            anchor_is_src = False
        else:
            continue   # seed–seed or no-seed row

        if anchor_kg is None or not nb_name:
            continue

        # Filter non-clinical vocabularies
        if nb_ntype not in _KEEP_VOCABS:
            stats["skipped_vocab"] += 1
            continue

        # Only add LOINC neighbours when the anchor is already a Diagnostic_Test.
        # LOINC in ClinGraph includes clinical observations (e.g. "Fever",
        # "Body temperature") that are findings, not diagnostic procedures —
        # pulling them in from symptom/condition anchors creates false Test nodes.
        anchor_type = G_enriched.nodes[anchor_kg].get("type", "")
        if nb_ntype == "LNC" and anchor_type != "Diagnostic_Test":
            stats["skipped_vocab"] += 1
            continue

        # Only add RXNORM/ATC neighbours when the anchor is a Treatment or Condition.
        if nb_ntype in ("RXNORM", "ATC") and anchor_type not in ("Treatment", "Condition"):
            stats["skipped_vocab"] += 1
            continue

        nb_code, _ = _parse_node_id(nb_id)
        node_type = _vocab_to_node_type(nb_ntype, nb_name, code=nb_code)
        prefix    = _NODE_TYPE_TO_PREFIX.get(node_type, "Symptom")
        new_kg_id = f"{prefix}: {nb_name.title()}"
        label_key = nb_name.lower()

        edge_attrs = dict(
            relationship="CLINGRAPH_RELATED",
            clingraph_relationship=relation,
            source="ClinGraph",
        )

        # Determine edge direction: preserve original ClinGraph direction
        def make_edge(node_a: str, node_b: str) -> None:
            edge_src, edge_tgt = (node_a, node_b) if anchor_is_src else (node_b, node_a)
            if not G_enriched.has_edge(edge_src, edge_tgt):
                G_enriched.add_edge(edge_src, edge_tgt, **edge_attrs)

        if label_key in existing_labels:
            existing_id = existing_labels[label_key]
            make_edge(anchor_kg, existing_id)
            stats["dedup_edges"] += 1
            continue

        # New node — populate standard vocab code attributes so the frontend
        # tooltip (which looks for icd10_code, snomed_ca_code, etc.) can display them.
        # nb_code was already parsed above for type classification.
        # ICD-10-CM (US) is stored separately from icd10_code (WHO) and
        # icd10ca_code (Canadian) to avoid conflating different ICD-10 variants.
        _VOCAB_TO_STD_ATTR = {
            "ICD10CM":    "icd10cm_code",
            "SNOMEDCT_US":"snomed_ca_code",
            "LNC":        "loinc_code",
            "RXNORM":     "rxcui",
            "ATC":        "atc_code",
        }
        std_attr = _VOCAB_TO_STD_ATTR.get(nb_ntype)
        node_attrs = dict(
            type=node_type,
            clingraph_node_id=nb_id,
            clingraph_vocab=nb_ntype,
            clingraph_code=nb_code,
            source="ClinGraph",
            synonyms=[],
        )
        if std_attr:
            node_attrs[std_attr] = nb_code
        G_enriched.add_node(new_kg_id, **node_attrs)
        existing_labels[label_key] = new_kg_id
        stats["new_nodes"] += 1

        make_edge(anchor_kg, new_kg_id)
        stats["new_edges"] += 1

    print(
        f"  Added {stats['new_nodes']} new nodes, "
        f"{stats['new_edges']} new edges, "
        f"{stats['dedup_edges']} edges to existing nodes."
    )
    if stats["skipped_vocab"]:
        print(f"  Skipped {stats['skipped_vocab']} edges with non-clinical vocabs (CPT/ICD9CM/PHECODE/UMLS_CUI).")
    return G_enriched


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Enrich the triage KG with ClinGraph 1-hop neighbourhood data."
    )
    parser.add_argument(
        "--base-pkl",
        default=os.path.join(script_dir, "triage_knowledge_graph.pkl"),
        help="Base triage KG pickle (default: triage_knowledge_graph.pkl)",
    )
    parser.add_argument(
        "--clingraph-dir",
        default=os.path.join(script_dir, "data", "clingraph_data"),
        help="Directory containing ClinGraph_nodes.csv and ClinGraph_edges.csv",
    )
    parser.add_argument(
        "--out-pkl",
        default=os.path.join(script_dir, "triage_knowledge_graph_enriched.pkl"),
        help="Output enriched KG pickle (default: triage_knowledge_graph_enriched.pkl)",
    )
    parser.add_argument(
        "--max-new-nodes",
        type=int,
        default=500,
        help="Maximum new nodes to add from ClinGraph (default: 500)",
    )
    args = parser.parse_args()

    node_csv = os.path.join(args.clingraph_dir, "ClinGraph_nodes.csv")
    edge_csv = os.path.join(args.clingraph_dir, "ClinGraph_edges.csv")

    # ── Validate inputs ──────────────────────────────────────────────────────
    missing_files = [
        (p, lbl) for p, lbl in [
            (args.base_pkl, "Base KG pkl"),
            (node_csv,      "ClinGraph_nodes.csv"),
            (edge_csv,      "ClinGraph_edges.csv"),
        ]
        if not os.path.exists(p)
    ]
    if missing_files:
        for path, lbl in missing_files:
            print(f"ERROR: {lbl} not found at: {path}", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: Load base KG ─────────────────────────────────────────────────
    print(f"\n[1/6] Loading base KG ...")
    with open(args.base_pkl, "rb") as f:
        G_base: nx.DiGraph = pickle.load(f)
    print(f"  Base KG: {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges.")

    # ── Step 2: Load ClinGraph ───────────────────────────────────────────────
    print("\n[2/6] Loading ClinGraph data ...")
    cg_nodes = load_clingraph_nodes(node_csv)
    cg_edges = load_clingraph_edges(edge_csv)

    # ── Step 3: Build code index ─────────────────────────────────────────────
    print("\n[3/6] Building vocabulary code index ...")
    code_index = build_code_index(cg_nodes)

    # ── Step 4: Match KG nodes to ClinGraph ─────────────────────────────────
    print("\n[4/6] Matching KG nodes to ClinGraph ...")
    kg_to_cg = match_kg_to_clingraph(G_base, code_index)

    if not kg_to_cg:
        print(
            "WARNING: No KG nodes matched to ClinGraph.\n"
            "  Ensure ontology codes (snomed_ca_code, loinc_code, icd10_code, rxcui)\n"
            "  are populated — re-run build_kg.py first if needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Step 5: Extract 1-hop neighbourhood ──────────────────────────────────
    print("\n[5/7] Extracting 1-hop ClinGraph neighbourhood ...")
    seed_ids = {cg_id for cg_ids in kg_to_cg.values() for cg_id in cg_ids}
    neighbourhood = extract_neighbourhood(cg_edges, seed_ids)

    # ── Step 6: Enrich existing node attributes ───────────────────────────────
    print("\n[6/7] Enriching existing node attributes from ClinGraph DEFINED edges ...")
    import copy
    G_base_enriched = copy.deepcopy(G_base)
    enrich_node_attributes(G_base_enriched, kg_to_cg, neighbourhood)

    # ── Step 7: Merge neighbourhood into KG ──────────────────────────────────
    print("\n[7/7] Merging ClinGraph neighbourhood into KG ...")
    G_enriched = merge_clingraph_neighbourhood(
        G_base_enriched, kg_to_cg, neighbourhood,
        max_new_nodes=args.max_new_nodes,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(args.out_pkl, "wb") as f:
        pickle.dump(G_enriched, f)

    n_base_n = G_base.number_of_nodes()
    n_base_e = G_base.number_of_edges()
    n_enr_n  = G_enriched.number_of_nodes()
    n_enr_e  = G_enriched.number_of_edges()

    print(f"\n{'─'*55}")
    print(f"  Base KG   :  {n_base_n:>5} nodes  {n_base_e:>5} edges")
    print(f"  ClinGraph :  {n_enr_n - n_base_n:>+5} nodes  {n_enr_e - n_base_e:>+5} edges")
    print(f"  Enriched  :  {n_enr_n:>5} nodes  {n_enr_e:>5} edges")
    print(f"{'─'*55}")
    print(f"  Saved → {args.out_pkl}")

    # ── Regression check ─────────────────────────────────────────────────────
    missing_edges = [(u, v) for u, v in G_base.edges() if not G_enriched.has_edge(u, v)]
    if missing_edges:
        print(f"\nWARNING: {len(missing_edges)} base edges missing from enriched graph!",
              file=sys.stderr)
    else:
        print("  Regression check: all base edges preserved. ✓")


if __name__ == "__main__":
    main()
