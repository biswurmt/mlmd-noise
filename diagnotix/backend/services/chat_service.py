"""chat_service.py — Nebius-backed clinical decision-support chat."""
from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from backend.models.schemas import ChatRequest

# Load repo root .env (three levels up from diagnotix/backend/services/)
load_dotenv(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")),
    override=False,
)

# ── Vector DB (RAG) setup ────────────────────────────────────────────────────
_VECTOR_DB_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge-graphs", "vector_db")
)
if _VECTOR_DB_DIR not in sys.path:
    sys.path.insert(0, _VECTOR_DB_DIR)

try:
    from build_vector_db import query_guidelines as _query_guidelines  # noqa: E402
    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical reasoning assistant working alongside emergency department \
clinicians — triage nurses and emergency physicians. You have access to a \
structured knowledge graph encoding triage rules, diagnostic pathways, and \
evidence weights derived from published guidelines and medical literature. \
You also have direct access to retrieved passages from a curated corpus of \
authoritative medical guidelines (AHA/ACC, ACR, NICE, ESC, ACEP, and others).

When responding, open with a brief, direct orientation to what the knowledge \
graph shows on the topic: the key relationships, notable co-occurrence counts \
or trial numbers, and what they imply. Keep this grounding tight — a few \
sentences at most, not a data dump. Then move into synthesis: bring your \
broader clinical knowledge to bear, reason through the presentation, and let \
the graph evidence surface naturally where it strengthens the argument.

Where the retrieved guideline passages support a specific claim, cite them \
inline using their bracket number — e.g. "per NICE guidance [2]" or \
"ACC/AHA recommend [1]". Do not quote passages at length; paraphrase or \
summarise the relevant point. At the end of your response, include a compact \
**References** section listing only the passages you actually cited, in the \
format: `[N] Source — Title`. Omit passages you did not use. If no passages \
were relevant, omit the References section entirely.

Use standard emergency medicine terminology and assume clinical fluency in your \
reader. Be concise. For lists of findings, differentials, or actions, use \
bullet points. Acknowledge genuine diagnostic uncertainty when it matters, and \
flag when specialist input or immediate stabilisation takes priority.

When naming a specific clinical concept drawn from the knowledge graph, use its \
exact label as it appears in the NODES section below — the plain label only, \
without any "Type:" prefix.

## Current Knowledge Graph Context

{kg_context}

{rag_section}\
"""

# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def _retrieve_rag_section(query: str, n_results: int = 4) -> str:
    """Query the Qdrant vector DB and return a formatted guideline-passages block.

    Uses the user's message as the semantic query with no test filter — results
    span all indexed guidelines so the chat is not restricted to the 5 tagged tests.
    Returns an empty string when the vector DB is unavailable.
    """
    if not _RAG_AVAILABLE:
        return ""
    try:
        hits = _query_guidelines(query, n_results=n_results)
        if not hits:
            return ""

        lines = ["## Retrieved Guideline Passages\n"]
        for i, h in enumerate(hits, 1):
            lines.append(
                f"[{i}] {h['source']} — {h['title']}\n"
                f"{h['context']}\n"
            )
        return "\n".join(lines)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# KG serialisation
# ---------------------------------------------------------------------------

def _serialize_kg(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], pathway: str | None) -> str:
    lines: list[str] = []

    pathway_display = pathway if pathway else "All Pathways"
    lines.append(f"PATHWAY: {pathway_display}")
    lines.append("")
    lines.append(f"NODES ({len(nodes)}):")

    for n in nodes:
        node_id: str = n.get("id", "")
        typ: str = n.get("node_type") or n.get("type") or "Unknown"
        label = node_id.split(": ", 1)[-1] if ": " in node_id else node_id

        # Build inline annotations
        annotations: list[str] = []
        if n.get("loinc_code"):
            annotations.append(f"LOINC: {n['loinc_code']}")
        if n.get("icd10_code"):
            annotations.append(f"ICD-10: {n['icd10_code']}")
        if n.get("ebi_open_code"):
            annotations.append(f"HP/MONDO: {n['ebi_open_code']}")
        if n.get("snomed_ca_code"):
            annotations.append(f"SNOMED: {n['snomed_ca_code']}")
        if n.get("guideline_source"):
            annotations.append(f"Source: {n['guideline_source']}")
        if n.get("trial_count"):
            annotations.append(f"trials: {n['trial_count']}")

        # Co-occurrence articles for this pathway
        test_evidence = n.get("test_evidence") or []
        if pathway:
            for ev in test_evidence:
                if ev.get("test") == pathway and ev.get("articles"):
                    annotations.append(f"articles: {ev['articles']}")

        ann_str = f"  ({' | '.join(annotations)})" if annotations else ""
        lines.append(f"[{typ}] {label}{ann_str}")

        # Synonyms on the next line
        syns = n.get("synonyms") or []
        if syns:
            lines.append(f"  Synonyms: {', '.join(str(s) for s in syns)}")

    lines.append("")
    lines.append(f"RELATIONSHIPS ({len(edges)}):")

    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        rel = e.get("relationship", "RELATED")
        annotations: list[str] = []
        if e.get("literature_weight"):
            annotations.append(f"articles: {e['literature_weight']}")
        if e.get("test_literature_weight"):
            annotations.append(f"test articles: {e['test_literature_weight']}")
        if e.get("trial_count"):
            annotations.append(f"trials: {e['trial_count']}")
        if e.get("guideline_source"):
            annotations.append(f"source: {e['guideline_source']}")
        ann_str = f"  ({', '.join(annotations)})" if annotations else ""
        lines.append(f"{src} → {rel} → {tgt}{ann_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini call (synchronous — run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def sync_chat(req: ChatRequest) -> str:
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("NEBIUS_API_KEY not set")

    kg_context = _serialize_kg(req.context.nodes, req.context.edges, req.context.pathway)
    rag_section = _retrieve_rag_section(req.message)
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        kg_context=kg_context,
        rag_section=rag_section,
    )

    client = OpenAI(
        base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
        api_key=api_key,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in req.history],
        {"role": "user", "content": req.message},
    ]

    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2.5-fast",
        messages=messages,
    )
    return response.choices[0].message.content
