"""chat_service.py — Nebius-backed clinical decision-support chat."""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from backend.models.schemas import ChatRequest

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical reasoning assistant working alongside emergency department \
clinicians — triage nurses and emergency physicians. You have access to a \
structured knowledge graph encoding triage rules, diagnostic pathways, and \
evidence weights derived from published guidelines and medical literature.

When responding, open with a brief, direct orientation to what the knowledge \
graph shows on the topic: the key relationships, notable co-occurrence counts \
or trial numbers, and what they imply. Keep this grounding tight — a few \
sentences at most, not a data dump. Then move into synthesis: bring your \
broader clinical knowledge to bear, reason through the presentation, and let \
the graph evidence surface naturally where it strengthens the argument. The \
goal is a response that reads like a knowledgeable colleague thinking aloud, \
not a structured report citing sources. Quantitative evidence from the graph \
(article counts, trial numbers, guideline sources) should feel like supporting \
colour, not scaffolding.

Use standard emergency medicine terminology and assume clinical fluency in your \
reader. Be concise. For lists of findings, differentials, or actions, use \
bullet points. Acknowledge genuine diagnostic uncertainty when it matters, and \
flag when specialist input or immediate stabilisation takes priority.

When naming a specific clinical concept drawn from the knowledge graph, use its \
exact label as it appears in the NODES section below — the plain label only, \
without any "Type:" prefix.

## Current Knowledge Graph Context

{kg_context}
"""

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
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(kg_context=kg_context)

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
