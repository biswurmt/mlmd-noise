"""chat_service.py — Nebius-backed clinical decision-support chat."""
from __future__ import annotations

import json
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

try:
    from backend.services.semantic_scholar import search_papers as _ss_search, format_abstracts_section as _ss_format
    _SS_AVAILABLE = True
except Exception:
    _SS_AVAILABLE = False


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical reasoning assistant working alongside emergency department \
clinicians — triage nurses and emergency physicians. You have access to a \
structured knowledge graph encoding triage rules, diagnostic pathways, and \
evidence weights derived from published guidelines and medical literature. \
You also have direct access to two retrieved evidence sources: (1) passages from \
a curated corpus of authoritative medical guidelines (AHA/ACC, ACR, NICE, ESC, \
ACEP, and others), and (2) research paper abstracts retrieved live from Semantic \
Scholar, providing recent peer-reviewed literature directly relevant to the query.

When responding, open with a brief, direct orientation to what the knowledge \
graph shows on the topic: the key relationships, notable co-occurrence counts \
or trial numbers, and what they imply. Keep this grounding tight — a few \
sentences at most, not a data dump. Then move into synthesis: bring your \
broader clinical knowledge to bear, reason through the presentation, and let \
the graph evidence surface naturally where it strengthens the argument.

You MUST actively use the retrieved guideline passages and research abstracts \
provided below — do not ignore them or treat them as optional context. For every \
specific clinical claim, recommendation, threshold, or management step you make, \
check whether a retrieved source supports it and cite it. Cite inline using the \
bracket number — e.g. "per NICE guidance [2]", "ACC/AHA recommend [1]", \
"as shown in [5]". Do not quote passages at length; paraphrase or summarise the \
relevant point. If a retrieved paper directly contradicts or refines a guideline, \
call that out explicitly. At the end of your response, include a **References** \
section listing every source you cited. \
Format guideline passages as: `[N] Source — Title`. \
Format research papers as: `[N] Authors (Year) — Title. <URL>`. \
Omit sources you did not cite. If no retrieved sources were relevant to any claim \
in your response, state that explicitly rather than omitting the section silently.

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
{scholar_section}\
"""

# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def _retrieve_semantic_scholar_section(
    queries: list[str], pathway: str | None, start_index: int = 1
) -> str:
    """Search Semantic Scholar for papers relevant to each query and pathway.

    Runs one search per query, deduplicates results by s2_id, and returns a
    single formatted abstracts block. Returns "" on failure / when unavailable.
    """
    if not _SS_AVAILABLE:
        return ""
    try:
        seen_ids: set[str] = set()
        all_papers: list[dict] = []
        for query in queries:
            search_query = f"{pathway} {query}" if pathway else query
            for paper in _ss_search(search_query, limit=2):
                s2_id = paper.get("s2_id") or paper.get("title", "")
                if s2_id not in seen_ids:
                    seen_ids.add(s2_id)
                    all_papers.append(paper)
        return _ss_format(all_papers, start_index=start_index)
    except Exception:
        return ""


def _retrieve_rag_section(queries: list[str], n_results_per_query: int = 3) -> str:
    """Query the Qdrant vector DB for each query and return a deduplicated guideline-passages block.

    Runs one search per query, deduplicates hits by (doc_id, chunk_index), and
    formats all unique results into a single numbered block.
    Returns an empty string when the vector DB is unavailable.
    """
    if not _RAG_AVAILABLE:
        return ""
    try:
        seen: set[tuple] = set()
        all_hits: list[dict] = []
        for query in queries:
            for h in _query_guidelines(query, n_results=n_results_per_query):
                key = (h.get("doc_id"), h.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    all_hits.append(h)
        if not all_hits:
            return ""
        lines = ["## Retrieved Guideline Passages\n"]
        for i, h in enumerate(all_hits, 1):
            lines.append(
                f"[{i}] {h['source']} — {h['title']}\n"
                f"{h['context']}\n"
            )
        return "\n".join(lines)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------

def _build_retrieval_queries(req: ChatRequest, client: OpenAI) -> list[str]:
    """Rewrite the user's message into one or more focused clinical search queries.

    Returns a list of query strings — one per distinct concept in the message.
    Multiple queries are issued when the user asks about several topics at once,
    allowing separate vector DB and Semantic Scholar searches per concept.
    Falls back to [req.message] on any error so RAG is never blocked.
    """
    try:
        n_turns = 6 if req.context.pathway else 3
        recent_turns = req.history[-n_turns:] if req.history else []
        history_block = "\n".join(
            f"[{m.role}]: {m.content[:200]}" for m in recent_turns
        )
        pathway_hint = (
            f"Active clinical pathway: {req.context.pathway}.\n" if req.context.pathway else ""
        )
        pathway_filter_instruction = (
            f"Only incorporate conversation context relevant to the "
            f"'{req.context.pathway}' clinical pathway. Ignore turns about unrelated topics."
            if req.context.pathway else
            "Incorporate relevant context from the conversation history."
        )
        conversation_block = f"Conversation:\n{history_block}\n\n" if history_block else ""
        user_content = (
            f"{pathway_hint}"
            f"{conversation_block}"
            f"Current question: {req.message}\n\n"
            "Clinical search queries (JSON array):"
        )
        resp = client.chat.completions.create(
            model="moonshotai/Kimi-K2.5-fast",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical query translator for a clinical guideline "
                        "retrieval system. Given a conversation fragment and a current "
                        "user question, identify each distinct clinical concept being asked "
                        "about and output one concise, specific search query per concept in "
                        "formal medical terminology. Each query should be self-contained "
                        "(no pronouns referring to prior turns), 1 sentence, dense with "
                        "medical terms. Output ONLY a JSON array of strings — no explanation, "
                        "no markdown. Maximum 3 queries. If the question covers a single "
                        "concept, return a single-element array. "
                        "Example: [\"syncope aetiology cardiac arrhythmia\", "
                        "\"vasovagal syncope diagnosis criteria\"]. "
                        f"{pathway_filter_instruction}"
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=2048,
            temperature=0,
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        finish_reason = choice.finish_reason
        print("[retrieval] rewrite LLM raw response:", repr(raw), "| finish_reason:", finish_reason)
        if not raw.strip():
            print("[retrieval] rewrite returned empty response — falling back to raw message")
            return [req.message]
        # Parse the JSON array; strip markdown code fences if the model adds them
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        queries = json.loads(cleaned)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            queries = [q.strip() for q in queries if q.strip()]
            if queries:
                print("[retrieval] rewritten queries:", queries)
                return queries
        print("[retrieval] rewrite returned unexpected format — falling back to raw message")
        return [req.message]
    except Exception as e:
        print("[retrieval] rewrite failed, falling back to raw message. Error:", e)
        return [req.message]


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

    client = OpenAI(
        base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
        api_key=api_key,
    )

    kg_context = _serialize_kg(req.context.nodes, req.context.edges, req.context.pathway)
    retrieval_queries = _build_retrieval_queries(req, client)
    print("[retrieval] queries sent to RAG:", retrieval_queries)
    rag_section = _retrieve_rag_section(retrieval_queries)
    print("[rag] raw section:\n", rag_section or "<empty>")
    # Continue citation numbering after RAG passages so bracket numbers are unique
    rag_count = rag_section.count("\n[") + (1 if rag_section.startswith("[") else 0)
    print("[retrieval] queries sent to Semantic Scholar:", retrieval_queries)
    scholar_section = _retrieve_semantic_scholar_section(
        retrieval_queries, req.context.pathway, start_index=rag_count + 1
    )
    print("[semantic_scholar] section:\n", scholar_section or "<empty>")
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        kg_context=kg_context,
        rag_section=rag_section,
        scholar_section=("\n" + scholar_section) if scholar_section else "",
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
