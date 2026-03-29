"""semantic_scholar.py — Semantic Scholar Academic Graph API client.

Provides keyword search and direct paper lookup with abstract retrieval.
Results are cached in-memory to avoid redundant requests within a session.

No API key is required for basic use (~100 req / 5 min).
Set SEMANTIC_SCHOLAR_API_KEY in .env to raise the rate limit.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv

load_dotenv(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")),
    override=False,
)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,year,authors,citationCount,externalIds"
_ABSTRACT_MAX_CHARS = 400
_REQUEST_DELAY = 1.1  # seconds between requests (free tier: ~1 req/s)

_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
_HEADERS = {"x-api-key": _api_key} if _api_key else {}

_last_request_time: float = 0.0


def _throttle() -> None:
    """Enforce minimum delay between requests."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _REQUEST_DELAY:
        time.sleep(_REQUEST_DELAY - elapsed)
    _last_request_time = time.monotonic()


def _build_paper_dict(data: dict) -> dict | None:
    """Convert a raw API paper object into a clean dict. Returns None if no abstract."""
    abstract = (data.get("abstract") or "").strip()
    if not abstract:
        return None

    external_ids = data.get("externalIds") or {}
    s2_id = data.get("paperId", "")
    authors = data.get("authors") or []
    if authors:
        first_author = authors[0].get("name", "")
        author_str = f"{first_author} et al." if len(authors) > 1 else first_author
    else:
        author_str = "Unknown"

    return {
        "title": data.get("title") or "Untitled",
        "abstract": abstract,
        "year": data.get("year"),
        "citation_count": data.get("citationCount", 0),
        "doi": external_ids.get("DOI"),
        "s2_id": s2_id,
        "authors": author_str,
        "url": f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else None,
    }


@lru_cache(maxsize=128)
def search_papers(query: str, limit: int = 3) -> tuple[dict, ...]:
    """Search Semantic Scholar by keyword query.

    Returns a tuple (hashable for lru_cache) of up to `limit` paper dicts,
    each containing: title, abstract, year, citation_count, doi, s2_id, authors, url.
    Papers without abstracts are skipped. Returns empty tuple on failure.
    """
    _throttle()
    try:
        resp = requests.get(
            f"{_BASE_URL}/paper/search",
            params={"query": query, "fields": _FIELDS, "limit": limit + 3},  # fetch extra to account for no-abstract skips
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        papers = []
        for item in resp.json().get("data", []):
            p = _build_paper_dict(item)
            if p:
                papers.append(p)
            if len(papers) >= limit:
                break
        return tuple(papers)
    except Exception as e:
        print(f"  [semantic_scholar] search_papers('{query}') failed: {e}")
        return ()


def get_paper_by_id(paper_id: str) -> dict | None:
    """Fetch a single paper by any Semantic Scholar identifier.

    Accepts: bare S2 paper ID, DOI:<doi>, ARXIV:<id>, MAG:<id>, ACL:<id>, DBLP:<id>.
    Returns a paper dict or None on failure / no abstract.
    """
    _throttle()
    try:
        resp = requests.get(
            f"{_BASE_URL}/paper/{paper_id}",
            params={"fields": _FIELDS},
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return _build_paper_dict(resp.json())
    except Exception as e:
        print(f"  [semantic_scholar] get_paper_by_id('{paper_id}') failed: {e}")
        return None


def format_abstracts_section(
    papers: list[dict] | tuple[dict, ...],
    heading: str = "Supporting Research Abstracts",
    start_index: int = 1,
) -> str:
    """Format a list of paper dicts as a numbered block for LLM prompt injection.

    Each entry includes title, authors, year, citation count, truncated abstract,
    and a Semantic Scholar URL. Returns empty string if papers is empty.
    `start_index` lets callers continue numbering after RAG passage citations.
    """
    if not papers:
        return ""

    lines = [f"## {heading}\n"]
    for i, p in enumerate(papers, start_index):
        year_str = f" ({p['year']})" if p.get("year") else ""
        citations = f"  |  {p['citation_count']} citations" if p.get("citation_count") else ""
        abstract = p["abstract"]
        if len(abstract) > _ABSTRACT_MAX_CHARS:
            abstract = abstract[:_ABSTRACT_MAX_CHARS].rstrip() + "…"
        url_line = f"\n  URL: {p['url']}" if p.get("url") else ""

        lines.append(
            f"[{i}] {p['authors']}{year_str} — {p['title']}{citations}\n"
            f"{abstract}"
            f"{url_line}\n"
        )
    return "\n".join(lines)
