"""kg_audit_agent.py
===================
ED-specific multi-agent validation pipeline for the triage knowledge graph.

Audits compiled PKL graph edges (not guideline_rules.json) using four distinct
clinical personas, then enriches each edge with audit metadata in-place.

Four Personas
-------------
  A — Protocol & Triage Enforcer  (Qdrant vector DB + LLM)
  B — Rapid Diagnostics Researcher (Semantic Scholar / Europe PMC; no LLM)
  C — ED Attending                 (pure LLM, no external tools)
  D — ED Medical Director          (LLM synthesis of A + B + C)

Writes
------
  • triage_knowledge_graph_enriched.pkl  — edge attributes enriched with audit metadata
  • audit_report.json                    — full per-triad report + summary
  • .audit_agent_cache.json              — checkpoint; resume with --resume

Usage
-----
  python kg_audit_agent.py
  python kg_audit_agent.py --dry-run
  python kg_audit_agent.py --acuity HIGH
  python kg_audit_agent.py --resume
  python kg_audit_agent.py --kg-path triage_knowledge_graph.pkl --threshold 0.6
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).parent.resolve()
load_dotenv(_THIS_DIR / ".env")
load_dotenv(_THIS_DIR.parent / ".env", override=False)

_provider   = os.environ.get("LLM_PROVIDER", "azure").lower()
_deployment = None
_CLIENT     = None

if _provider == "nebius":
    from openai import OpenAI as _OpenAI
    _nebius_key = os.environ.get("NEBIUS_API_KEY")
    if not _nebius_key:
        raise ValueError("NEBIUS_API_KEY not set — required when LLM_PROVIDER=nebius.")
    _deployment = os.environ.get(
        "NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-fast"
    )
    _CLIENT = _OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=_nebius_key,
    )
else:  # azure (default)
    from azure.identity import AzureCliCredential, get_bearer_token_provider
    from openai import AzureOpenAI
    _endpoint   = os.environ.get("ENDPOINT_URL")
    _deployment = os.environ.get("DEPLOYMENT_NAME")
    if not _endpoint or not _deployment:
        raise ValueError("ENDPOINT_URL / DEPLOYMENT_NAME missing from .env")
    _credential     = AzureCliCredential()
    _token_provider = get_bearer_token_provider(
        _credential, "https://cognitiveservices.azure.com/.default"
    )
    _CLIENT = AzureOpenAI(
        azure_endpoint=_endpoint,
        azure_ad_token_provider=_token_provider,
        api_version="2025-01-01-preview",
    )

_SS_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

KG_DEFAULT_PATH = str(_THIS_DIR / "triage_knowledge_graph_enriched.pkl")
CACHE_PATH      = _THIS_DIR / ".audit_agent_cache.json"
REPORT_DEFAULT  = str(_THIS_DIR / "audit_report.json")

CONFIDENCE_THRESHOLD = 0.7

# Acuity mapping by canonical test label (matches node IDs: "Test: <label>")
_ACUITY = {
    "ECG":                   "HIGH",
    "CT Head":               "HIGH",
    "Testicular Ultrasound": "MODERATE",
    "Appendix Ultrasound":   "MODERATE",
    "Abdominal Ultrasound":  "MODERATE",
    "Arm X-Ray":             "STANDARD",
    "Chest X-Ray":           "STANDARD",
}
_ACUITY_ORDER = {"HIGH": 0, "MODERATE": 1, "STANDARD": 2}

# Keywords that signal a rule-out (exclusion) pathway
_RULE_OUT_KEYWORDS = (
    "rule out", "exclude", "exclusion", "negative predictive",
    "rule-out", "r/o", "to exclude",
)

# Semantic Scholar API
_SS_BASE = "https://api.semanticscholar.org/graph/v1"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: graph I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_graph(kg_path: str):
    """Load and return the NetworkX DiGraph from a pickle file."""
    with open(kg_path, "rb") as fh:
        return pickle.load(fh)


def _save_graph(G, kg_path: str) -> None:
    with open(kg_path, "wb") as fh:
        pickle.dump(G, fh)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"  [WARN] Could not write cache: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Sub-graph extraction by acuity
# ─────────────────────────────────────────────────────────────────────────────

def _is_rule_out(condition: str | None, relationship: str | None) -> bool:
    """Return True when the triad represents a test used to exclude a diagnosis."""
    text = " ".join(filter(None, [condition, relationship])).lower()
    return any(kw in text for kw in _RULE_OUT_KEYWORDS)


def _acuity_for_test(test_label: str) -> str:
    """Derive acuity bucket from the bare test name (strips 'Test: ' prefix)."""
    bare = test_label.removeprefix("Test: ").strip()
    return _ACUITY.get(bare, "STANDARD")


def _test_slug(test_label: str) -> str:
    """Convert 'Test: Arm X-Ray' → 'arm_xray' for Qdrant filter."""
    bare = test_label.removeprefix("Test: ").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", bare).strip("_")


def _extract_triads(G) -> list[dict]:
    """Traverse the graph and return a sorted, deduplicated list of audit triads.

    Deduplication is by (finding, test) across both path types. The 3-hop path
    (finding → condition → test) is preferred over the direct 2-hop shortcut
    because it carries richer condition context for the personas. A direct triad
    is only emitted when no 3-hop triad exists for that (finding, test) pair.

    Each triad is a dict with keys:
        finding, condition (str|None), test, source, acuity, rule_out, cache_key
    """
    # ── Pass 1: collect all 3-hop triads ────────────────────────────────────
    seen_ft: set[tuple] = set()   # (finding, test) — dedup key across path types
    seen_3hop: set[tuple] = set() # (finding, condition, test) — within-pass dedup
    triads: list[dict] = []

    for src, tgt, data in G.edges(data=True):
        if data.get("relationship") != "INDICATES_CONDITION":
            continue
        condition_node = tgt
        for _, test_node, test_data in G.out_edges(condition_node, data=True):
            if test_data.get("relationship") != "REQUIRES_TEST":
                continue
            key3 = (src, condition_node, test_node)
            if key3 in seen_3hop:
                continue
            seen_3hop.add(key3)
            seen_ft.add((src, test_node))
            triads.append({
                "finding":   src,
                "condition": condition_node,
                "test":      test_node,
                "source":    data.get("source", ""),
                "acuity":    _acuity_for_test(test_node),
                "rule_out":  _is_rule_out(condition_node, None),
                "cache_key": f"{src}|{condition_node}|{test_node}",
            })

    # ── Pass 2: direct edges only when no 3-hop covers the same (finding, test)
    seen_direct: set[tuple] = set()
    for src, tgt, data in G.edges(data=True):
        if data.get("relationship") != "DIRECTLY_INDICATES_TEST":
            continue
        ft_key = (src, tgt)
        if ft_key in seen_ft or ft_key in seen_direct:
            continue
        seen_direct.add(ft_key)
        triads.append({
            "finding":   src,
            "condition": None,
            "test":      tgt,
            "source":    data.get("source", ""),
            "acuity":    _acuity_for_test(tgt),
            "rule_out":  _is_rule_out(None, None),
            "cache_key": f"{src}||{tgt}",
        })

    triads.sort(key=lambda t: _ACUITY_ORDER.get(t["acuity"], 2))
    return triads


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(system_msg: str, user_msg: str, retries: int = 2) -> str:
    """Call the configured LLM and return the raw text response."""
    for attempt in range(retries + 1):
        try:
            response = _CLIENT.chat.completions.create(
                model=_deployment,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise exc
    return ""


def _parse_json_response(text: str) -> dict | None:
    """Parse a JSON dict from a possibly-fenced LLM response."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Persona A — Protocol & Triage Enforcer
# ─────────────────────────────────────────────────────────────────────────────

_PERSONA_A_SYSTEM = """\
You are an Emergency Medicine QA Auditor. Review retrieved guideline chunks to
determine whether the stated clinical pathway is explicitly supported by an
authoritative ED guideline body.

Focus on:
1. Does an authoritative ED guideline body (ACEP, CAEP, NICE, AHA/ACC, CTAS, ACR)
   explicitly recommend or endorse this test for this presentation?
2. Does the pathway appropriately incorporate validated clinical decision rules
   (Wells Score, Ottawa Rules, HEART Score, PECARN, NEXUS, Canadian CT Head Rule)
   before ordering the test, when applicable?
3. Does any "Choosing Wisely" initiative flag this test as potentially overused in ED?

Return ONLY valid JSON — no markdown fences, no explanation:
{
  "support_level": "strong | moderate | weak | none",
  "matched_guideline": "<body + section if found, else null>",
  "decision_rule_required": "<rule name if applicable, else null>",
  "choosing_wisely_flag": true | false,
  "excerpt": "<most relevant 1-2 sentence quote from retrieved chunks, or null>"
}"""


def _persona_a(triad: dict) -> dict:
    """Persona A: query Qdrant for guideline support, interpret with LLM."""
    finding   = triad["finding"].split(": ", 1)[-1]
    condition = triad["condition"].split(": ", 1)[-1] if triad["condition"] else None
    test      = triad["test"].split(": ", 1)[-1]
    source    = triad["source"]

    # Graceful import — Qdrant/Nebius may not be configured in all envs
    chunks_text = "(Qdrant not available — proceeding without vector context)"
    try:
        sys.path.insert(0, str(_THIS_DIR / "vector_db"))
        from build_vector_db import query_guidelines  # noqa: PLC0415
        query = f'{source} guidelines "{finding}" "{test}" emergency department'
        results = query_guidelines(query, n_results=5, diagnostic_test=_test_slug(triad["test"]))
        if results:
            parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                src   = r.get("source", "")
                score = r.get("score", 0)
                ctx   = (r.get("context") or r.get("matched_text", ""))[:600]
                parts.append(f"[{i}] {title} ({src}, score={score:.2f})\n{ctx}")
            chunks_text = "\n\n".join(parts)
        else:
            chunks_text = "(No relevant guideline chunks retrieved from vector DB)"
    except Exception as exc:
        chunks_text = f"(Vector DB unavailable: {exc})"

    pathway = f"[{finding}] → [{condition or 'direct'}] → [{test}]"
    user_msg = (
        f"Pathway: {pathway}\n"
        f"Guideline source cited in KG: {source}\n"
        f"Rule-out pathway: {triad['rule_out']}\n\n"
        f"Retrieved guideline chunks:\n{chunks_text}"
    )

    raw = _call_llm(_PERSONA_A_SYSTEM, user_msg)
    result = _parse_json_response(raw)
    if not isinstance(result, dict):
        result = {
            "support_level": "none",
            "matched_guideline": None,
            "decision_rule_required": None,
            "choosing_wisely_flag": False,
            "excerpt": None,
            "_parse_error": raw[:200],
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Persona B — Rapid Diagnostics Researcher  (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

_POC_KEYWORDS = ("point-of-care", "point of care", "pocus", "bedside ultrasound",
                 "rapid test", "poc test")
_NPV_KEYWORDS = ("negative predictive value", "npv", "sensitivity", "rule out")


def _semantic_scholar_search(query: str, limit: int = 10) -> list[dict]:
    """Search Semantic Scholar. Returns list of paper dicts (title, year, abstract)."""
    headers = {"x-api-key": _SS_API_KEY} if _SS_API_KEY else {}
    try:
        resp = requests.get(
            f"{_SS_BASE}/paper/search",
            params={"query": query, "fields": "title,year,abstract", "limit": limit},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


def _pmc_search(term1: str, term2: str) -> int:
    """Return Europe PMC hitCount for articles co-mentioning term1 and term2."""
    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query":    f"({term1}) AND ({term2})",
                "format":   "json",
                "pageSize": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("hitCount", 0)
    except Exception:
        return 0


def _persona_b(triad: dict) -> dict:
    """Persona B: literature search via Semantic Scholar (PMC fallback)."""
    test_bare = triad["test"].split(": ", 1)[-1]
    cond_bare = (triad["condition"].split(": ", 1)[-1]
                 if triad["condition"] else test_bare)

    query     = f"{cond_bare} {test_bare} emergency department sensitivity"
    papers    = _semantic_scholar_search(query, limit=10)
    source_used = "semantic_scholar"

    if not papers:
        # Fallback to Europe PMC
        count = _pmc_search(cond_bare, test_bare)
        return {
            "paper_count":         count,
            "mean_year":           None,
            "recency_flag":        "unknown",
            "poc_alternative_found": False,
            "npv_mentioned":       False,
            "source_used":         "europepmc" if count else "none",
        }

    years = [p["year"] for p in papers if p.get("year")]
    mean_year = sum(years) / len(years) if years else None

    if mean_year is None:
        recency = "unknown"
    elif mean_year >= 2020:
        recency = "current"
    elif mean_year >= 2015:
        recency = "aging"
    else:
        recency = "outdated"

    abstracts_combined = " ".join(
        (p.get("abstract") or "").lower() for p in papers
    )
    poc_found = any(kw in abstracts_combined for kw in _POC_KEYWORDS)
    npv_found = any(kw in abstracts_combined for kw in _NPV_KEYWORDS)

    return {
        "paper_count":           len(papers),
        "mean_year":             round(mean_year, 1) if mean_year else None,
        "recency_flag":          recency,
        "poc_alternative_found": poc_found,
        "npv_mentioned":         npv_found,
        "source_used":           source_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persona C — ED Attending (Flow & Pragmatism)
# ─────────────────────────────────────────────────────────────────────────────

_PERSONA_C_SYSTEM = """\
You are a veteran Emergency Department Attending reviewing a diagnostic workflow
for appropriateness and patient flow. You have no access to external tools —
reason from your clinical knowledge only.

Evaluate whether this pathway reflects sound acute-care practice by checking:
1. Over-triaging: Is an expensive, invasive, or radioactive test ordered when a
   bedside assessment, clinical decision score, or cheaper alternative should
   precede it? (e.g. ordering CT before a Wells Score is calculated for PE)
2. Under-triaging: Does the pathway miss a critical red-flag symptom that mandates
   an urgent rule-out test to be ordered immediately?
3. Safe discharge: If the test result is negative, does the pathway logically
   support safe discharge, or does it leave a dangerous diagnostic gap?
4. Never-event risk: Could this pathway lead to a missed time-sensitive diagnosis
   (STEMI, acute stroke, testicular torsion, ectopic pregnancy, cord compression,
   necrotizing fasciitis)?

Return ONLY valid JSON — no markdown fences, no explanation:
{
  "logical": true | false,
  "over_testing_concern": true | false,
  "under_testing_concern": true | false,
  "red_flag_present": true | false,
  "never_event_risk": true | false,
  "concerns": ["<concise concern>"],
  "safer_first_step": "<cheaper or faster bedside alternative, or null>"
}"""


def _persona_c(triad: dict) -> dict:
    """Persona C: ED Attending clinical pragmatism — pure LLM, no tools."""
    finding   = triad["finding"].split(": ", 1)[-1]
    condition = triad["condition"].split(": ", 1)[-1] if triad["condition"] else None
    test      = triad["test"].split(": ", 1)[-1]

    user_msg = (
        f"Patient acuity: {triad['acuity']}  (HIGH | MODERATE | STANDARD)\n"
        f"Pathway: [{finding}] → [{condition or 'direct'}] → [{test}]\n"
        f"Rule-out pathway: {triad['rule_out']}\n"
        f"Guideline source: {triad['source']}"
    )

    raw    = _call_llm(_PERSONA_C_SYSTEM, user_msg)
    result = _parse_json_response(raw)
    if not isinstance(result, dict):
        result = {
            "logical": True,
            "over_testing_concern": False,
            "under_testing_concern": False,
            "red_flag_present": False,
            "never_event_risk": False,
            "concerns": [],
            "safer_first_step": None,
            "_parse_error": raw[:200],
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Persona D — ED Medical Director (Consensus Judge)
# ─────────────────────────────────────────────────────────────────────────────

_PERSONA_D_SYSTEM = """\
You are the ED Medical Director reviewing a KG pathway audit. You have received
three independent assessments from: a Protocol Enforcer (guideline compliance),
a Literature Researcher (evidence recency), and an ED Attending (clinical flow).

Synthesize them into a final verdict. Prioritize patient safety and rapid
rule-out of life-threatening conditions above all else.

Weighting guidance:
- Persona A (protocol): carries most weight. "strong" guideline support overrides
  minor clinical concerns. "none" support is a strong signal toward Flagged/Rejected.
- Persona B (literature): refines confidence. High paper count + current recency +
  NPV data → higher score. Zero papers → reduce confidence by 0.15–0.25.
- Persona C (clinical pragmatism): hard veto ONLY when never_event_risk=true, OR
  when both over_testing_concern=true AND support_level is "weak" or "none".

Hard-flag triggers — always output "Flagged for Review" regardless of score:
  • Persona A: choosing_wisely_flag = true
  • Persona C: never_event_risk = true
  • Persona B: poc_alternative_found = true when test is invasive or radioactive
  • Rule-out pathway where Persona A support_level is "weak" or "none"
    (a missed rule-out in the ED is a direct patient safety risk)

Confidence scoring guidance (0.0–1.0):
  • 0.85–1.00 : Verified — strong guideline support + current literature + no concerns
  • 0.70–0.84 : Verified — good evidence, minor concerns only
  • 0.50–0.69 : Flagged for Review — mixed evidence or process concern
  • 0.00–0.49 : Rejected — no guideline support, clinical concerns, or zero literature

Return ONLY valid JSON — no markdown fences, no explanation:
{
  "status": "Verified | Flagged for Review | Rejected",
  "confidence_score": 0.0,
  "hard_flag_reason": "<reason string if hard-flagged, else null>",
  "rationale": "<2-3 sentence synthesis>"
}"""


def _persona_d(triad: dict, pa: dict, pb: dict, pc: dict) -> dict:
    """Persona D: synthesise A + B + C into final verdict."""
    finding   = triad["finding"].split(": ", 1)[-1]
    condition = triad["condition"].split(": ", 1)[-1] if triad["condition"] else None
    test      = triad["test"].split(": ", 1)[-1]

    user_msg = (
        f"Pathway: [{finding}] → [{condition or 'direct'}] → [{test}]\n"
        f"Acuity: {triad['acuity']}  |  Rule-out: {triad['rule_out']}\n\n"
        f"--- Persona A (Protocol Enforcer) ---\n{json.dumps(pa, indent=2)}\n\n"
        f"--- Persona B (Literature Researcher) ---\n{json.dumps(pb, indent=2)}\n\n"
        f"--- Persona C (ED Attending) ---\n{json.dumps(pc, indent=2)}"
    )

    raw    = _call_llm(_PERSONA_D_SYSTEM, user_msg)
    result = _parse_json_response(raw)
    if not isinstance(result, dict):
        result = {
            "status": "Flagged for Review",
            "confidence_score": 0.5,
            "hard_flag_reason": "LLM parse failure — manual review required",
            "rationale": raw[:300],
        }
    # Ensure confidence_score is a float in [0, 1]
    try:
        result["confidence_score"] = max(0.0, min(1.0, float(result["confidence_score"])))
    except (TypeError, ValueError):
        result["confidence_score"] = 0.5
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-triad orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _audit_triad(triad: dict, cache: dict) -> dict:
    """Run personas A → B → C → D for one triad. Returns the full audit result dict."""
    ck = triad["cache_key"]
    if ck in cache:
        return cache[ck]

    print(
        f"  [{triad['acuity']}] {triad['finding'].split(': ',1)[-1]} → "
        f"{(triad['condition'] or '').split(': ',1)[-1] or 'direct'} → "
        f"{triad['test'].split(': ',1)[-1]}"
    )

    pa = _persona_a(triad)
    pb = _persona_b(triad)
    pc = _persona_c(triad)
    pd = _persona_d(triad, pa, pb, pc)

    result = {
        "finding":          triad["finding"],
        "condition":        triad["condition"],
        "test":             triad["test"],
        "acuity":           triad["acuity"],
        "rule_out":         triad["rule_out"],
        "source":           triad["source"],
        "status":           pd.get("status", "Flagged for Review"),
        "confidence_score": pd.get("confidence_score", 0.5),
        "hard_flag_reason": pd.get("hard_flag_reason"),
        "rationale":        pd.get("rationale", ""),
        "persona_a":        pa,
        "persona_b":        pb,
        "persona_c":        pc,
    }
    cache[ck] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Graph mutation
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_graph(G, results: list[dict]) -> None:
    """Write audit metadata back onto the corresponding graph edges."""
    # Build lookup: (finding, condition_or_None, test) → audit result
    lookup: dict[tuple, dict] = {}
    for r in results:
        key = (r["finding"], r["condition"], r["test"])
        lookup[key] = r

    now_iso = datetime.now(timezone.utc).isoformat()

    for src, tgt, data in G.edges(data=True):
        rel = data.get("relationship", "")

        if rel == "DIRECTLY_INDICATES_TEST":
            r = lookup.get((src, None, tgt))
        elif rel == "INDICATES_CONDITION":
            # Enrich the finding→condition edge using the triad where this finding
            # leads (via condition) to the test — pick the first matching result
            r = next(
                (v for v in results
                 if v["finding"] == src and v["condition"] == tgt),
                None,
            )
        elif rel == "REQUIRES_TEST":
            r = next(
                (v for v in results
                 if v["condition"] == src and v["test"] == tgt),
                None,
            )
        else:
            continue

        if r is None:
            continue

        data["audit_status"]       = r["status"]
        data["confidence_score"]   = r["confidence_score"]
        data["last_audited"]       = now_iso
        data["persona_a_support"]  = r["persona_a"].get("support_level", "none")
        data["persona_b_papers"]   = r["persona_b"].get("paper_count", 0)
        data["persona_c_concerns"] = r["persona_c"].get("concerns", [])
        data["hard_flag_reason"]   = r["hard_flag_reason"]
        data["audit_rationale"]    = r["rationale"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_report(results: list[dict], args) -> None:
    """Write audit_report.json and print a console summary."""
    verified  = [r for r in results if r["status"] == "Verified"]
    flagged   = [r for r in results if r["status"] == "Flagged for Review"]
    rejected  = [r for r in results if r["status"] == "Rejected"]
    hard_flag = [r for r in results if r.get("hard_flag_reason")]
    never_ev  = [r for r in results if r["persona_c"].get("never_event_risk")]

    total = len(results)
    mean_conf = (
        sum(r["confidence_score"] for r in results) / total if total else 0.0
    )

    report = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "kg_path":    args.kg_path,
        "threshold":  args.threshold,
        "dry_run":    args.dry_run,
        "summary": {
            "total_triads":    total,
            "verified":        len(verified),
            "flagged":         len(flagged),
            "rejected":        len(rejected),
            "mean_confidence": round(mean_conf, 3),
            "hard_flagged":    len(hard_flag),
            "never_event_risks": len(never_ev),
        },
        "triads": results,
    }

    if not args.dry_run:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written → {args.report}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"KG Audit Complete — {total} triads audited")
    print(f"  Verified:           {len(verified):3d}  ({100*len(verified)/max(total,1):.1f}%)"
          f"   mean confidence: "
          f"{sum(r['confidence_score'] for r in verified)/max(len(verified),1):.2f}")
    print(f"  Flagged for Review: {len(flagged):3d}  ({100*len(flagged)/max(total,1):.1f}%)"
          f"   mean confidence: "
          f"{sum(r['confidence_score'] for r in flagged)/max(len(flagged),1):.2f}")
    print(f"  Rejected:           {len(rejected):3d}  ({100*len(rejected)/max(total,1):.1f}%)"
          f"   mean confidence: "
          f"{sum(r['confidence_score'] for r in rejected)/max(len(rejected),1):.2f}")
    print(f"  Hard-flagged:       {len(hard_flag):3d}  (never-event risk or Choosing Wisely)")

    attention = flagged + rejected
    if attention:
        print(f"\nTriads requiring review (confidence < {args.threshold} or hard-flagged):")
        for r in sorted(attention, key=lambda x: x["confidence_score"]):
            finding   = r["finding"].split(": ", 1)[-1]
            condition = (r["condition"] or "").split(": ", 1)[-1] or "direct"
            test      = r["test"].split(": ", 1)[-1]
            flag_note = f"  !! {r['hard_flag_reason']}" if r.get("hard_flag_reason") else ""
            print(
                f"\n  [{r['acuity']}]  {finding} → {condition} → {test}"
                f"\n          Status: {r['status']}  (confidence: {r['confidence_score']:.2f})"
                f"\n          {r['rationale'][:120]}"
                + (f"\n{flag_note}" if flag_note else "")
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ED-specific multi-agent KG audit pipeline."
    )
    parser.add_argument(
        "--kg-path", default=KG_DEFAULT_PATH,
        help="Path to .pkl graph file (default: triage_knowledge_graph_enriched.pkl)",
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help="Flag triads below this confidence score (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only — do not write PKL or audit_report.json",
    )
    parser.add_argument(
        "--report", default=REPORT_DEFAULT,
        help="Path for the output JSON report",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load .audit_agent_cache.json and skip already-audited triads",
    )
    parser.add_argument(
        "--acuity", choices=["HIGH", "MODERATE", "STANDARD"],
        help="Audit only this acuity bucket (default: all)",
    )
    args = parser.parse_args()

    if not Path(args.kg_path).exists():
        print(f"[ERROR] KG file not found: {args.kg_path}")
        sys.exit(1)

    print(f"Loading graph from {args.kg_path} …")
    G = _load_graph(args.kg_path)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Extracting triads …")
    triads = _extract_triads(G)
    if args.acuity:
        triads = [t for t in triads if t["acuity"] == args.acuity]
    print(f"  {len(triads)} unique triads to audit (acuity filter: {args.acuity or 'all'})")

    cache: dict = _load_cache() if args.resume else {}
    cached_count = sum(1 for t in triads if t["cache_key"] in cache)
    if cached_count:
        print(f"  {cached_count} triads already in cache — skipping")

    print("\nRunning multi-agent audit …")
    results: list[dict] = []
    for i, triad in enumerate(triads, 1):
        print(f"\nTriad {i}/{len(triads)}", end=" ")
        result = _audit_triad(triad, cache)
        results.append(result)
        _save_cache(cache)  # checkpoint after every triad

    print("\nEnriching graph edges …")
    if not args.dry_run:
        _enrich_graph(G, results)
        _save_graph(G, args.kg_path)
        print(f"  Graph saved → {args.kg_path}")
    else:
        print("  (dry-run — graph not modified)")

    _generate_report(results, args)


if __name__ == "__main__":
    main()
