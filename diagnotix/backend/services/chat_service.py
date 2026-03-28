"""chat_service.py — Anthropic-backed clinical decision-support chat."""
from __future__ import annotations

import os
from typing import Any

import anthropic

from backend.models.schemas import ChatRequest

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical decision-support assistant embedded in Diagnotix, a medical \
knowledge graph tool used by emergency department clinicians — triage nurses and \
emergency physicians.

Your role is to help interpret the current triage knowledge graph, support \
clinical reasoning at the point of care, and provide evidence-based guidance \
grounded in published guidelines.

## Guidelines

1. **Ground responses in the knowledge graph.** Cite specific nodes, relationship \
types, co-occurrence counts (literature_articles, literature_patents), and \
trial_count values when they support your reasoning. Numbers from the graph are \
authoritative.

2. **Supplement with general medical knowledge** where the graph is silent, but \
flag it clearly (e.g. "Outside the current graph: …").

3. **Use standard clinical terminology** appropriate for emergency medicine \
colleagues. Assume working knowledge of vitals, triage protocols, and common \
diagnostic procedures.

4. **Cite nodes by their exact label.** When referencing a specific node from the \
knowledge graph, use its exact label name as it appears in the "NODES" section \
below (e.g. "Chest Pain", "STEMI", "Abdominal Ultrasound"). Do **not** use the \
"Type: Label" prefix format — just the label. Only reference labels that appear \
verbatim in the graph context below.

5. **Be concise and actionable.** Lead with the most clinically important \
information. Use bullet points for differentials, test lists, or findings. \
Emergency clinicians need fast answers.

6. **Acknowledge uncertainty.** Recommend specialist escalation when appropriate. \
This tool supports — never replaces — clinical judgment and institutional protocols.

7. **For critical presentations**, prioritise stabilisation before further workup.

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
# Anthropic call (synchronous — run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def sync_chat(req: ChatRequest) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    kg_context = _serialize_kg(req.context.nodes, req.context.edges, req.context.pathway)
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(kg_context=kg_context)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            *[{"role": m.role, "content": m.content} for m in req.history],
            {"role": "user", "content": req.message},
        ],
    )
    return response.content[0].text  # type: ignore[index]
