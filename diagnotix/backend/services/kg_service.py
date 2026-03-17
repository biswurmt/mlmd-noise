"""kg_service.py
================
Orchestrates the Diagnotix add-test pipeline:
  1. Snapshot the current graph node IDs (for diff computation).
  2. Call Azure OpenAI to extract triage rules for the requested diagnostic test.
  3. Validate and append the new rules to guideline_rules.json.
  4. Invoke generate_knowledge_graph() from build_kg.py to rebuild the PKL.
  5. Compute the diff (new nodes / new edges) against the pre-rebuild snapshot.
  6. Return an AddTestResponse with the diff so the frontend can append only
     the new cluster to the existing visualisation.
"""

import asyncio
import json
import os
import re
import sys

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from backend.models.schemas import AddTestResponse
from backend.services.graph_service import get_existing_node_ids, load_graph_json

# ─────────────────────────────────────────────────────────────────────────────
# Resolve the knowledge-graphs directory so build_kg can be imported.
# ─────────────────────────────────────────────────────────────────────────────
_KG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge-graphs")
)
if _KG_DIR not in sys.path:
    sys.path.insert(0, _KG_DIR)

from build_kg import generate_knowledge_graph  # noqa: E402

_RULES_FILE = os.path.join(_KG_DIR, "guideline_rules.json")

# ─────────────────────────────────────────────────────────────────────────────
# Azure OpenAI client — uses AzureCliCredential (no API key required).
# Set ENDPOINT_URL and DEPLOYMENT_NAME in knowledge-graphs/.env.
# ─────────────────────────────────────────────────────────────────────────────
_endpoint = os.environ.get("ENDPOINT_URL")
_deployment = os.environ.get("DEPLOYMENT_NAME")

if not _endpoint:
    raise ValueError(
        "ENDPOINT_URL not found in the environment. "
        "Add it to knowledge-graphs/.env and restart the server."
    )
if not _deployment:
    raise ValueError(
        "DEPLOYMENT_NAME not found in the environment. "
        "Add it to knowledge-graphs/.env and restart the server."
    )

_credential = AzureCliCredential()
_token_provider = get_bearer_token_provider(
    _credential, "https://cognitiveservices.azure.com/.default"
)

_CLIENT = AzureOpenAI(
    azure_endpoint=_endpoint,
    azure_ad_token_provider=_token_provider,
    api_version="2025-01-01-preview",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
VALID_NODE_TYPES = {
    "Symptom",
    "Vital_Sign_Threshold",
    "Demographic_Factor",
    "Risk_Factor",
    "Clinical_Attribute",
    "Mechanism_of_Injury",
}

SYSTEM_PROMPT = """\
You are a clinical knowledge engineer that populates a medical triage knowledge graph.
Given a diagnostic test name, generate triage rules following the EXACT schema below.

OUTPUT FORMAT: A valid JSON array and nothing else — no markdown fences, no explanation.

Each rule is a JSON object with these fields:
  "raw_symptom"  (required) — clinical finding, vital sign threshold, risk factor, or
                              demographic/anatomical factor that warrants this test
                              (e.g. "chest pain", "heart rate > 100", "diabetes mellitus")
  "node_type"    (required) — MUST be exactly one of:
                              "Symptom" | "Vital_Sign_Threshold" | "Demographic_Factor" |
                              "Risk_Factor" | "Clinical_Attribute" | "Mechanism_of_Injury"
  "condition"    (required) — medical condition the symptom/finding indicates
                              (e.g. "Pulmonary Embolism", "Subdural Hematoma")
  "test"         (required) — MUST be the exact diagnostic test string provided
  "source"       (required) — real guideline abbreviation
                              (e.g. "ACR", "AHA/ACC", "NICE", "ESC", "ACEP", "WHO", "IDSA")
  "treatment"    (optional) — include ONLY when there is a well-established first-line
                              treatment (e.g. "Aspirin", "Alteplase")

Generate 8–15 rules spanning at least 3 distinct conditions. \
Use real clinical guideline sources. Be clinically accurate.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous worker (runs in a thread-pool via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────────

def _sync_add_test(diagnostic_test: str) -> AddTestResponse:
    # ── 1. Snapshot existing graph node IDs before rebuild ───────────────────
    existing_node_ids = get_existing_node_ids()

    # ── 2. LLM extraction ────────────────────────────────────────────────────
    user_message = (
        f'Generate triage rules for the diagnostic test: "{diagnostic_test}"\n\n'
        f"Cover at least 3–4 different conditions this test can diagnose and include "
        f"a mix of node types (Symptom, Vital_Sign_Threshold, Risk_Factor, etc.).\n\n"
        f'IMPORTANT: the "test" field must be exactly "{diagnostic_test}" for every rule.'
    )

    response = _CLIENT.chat.completions.create(
        model=_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # response_format={"type": "json_object"}, # commented out because the older model version
    )

    text = response.choices[0].message.content
    if not text:
        raise ValueError("Azure OpenAI returned no text content.")

    # ── 3. Parse JSON ────────────────────────────────────────────────────────
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError(
                f"Azure OpenAI response could not be parsed as JSON. "
                f"First 300 chars: {text[:300]}"
            )

    # json_object mode may wrap the array under a key (e.g. {"rules": [...]})
    if isinstance(parsed, dict):
        new_rules = next(
            (v for v in parsed.values() if isinstance(v, list)), None
        )
        if new_rules is None:
            raise ValueError("Azure OpenAI response JSON object contained no array.")
    elif isinstance(parsed, list):
        new_rules = parsed
    else:
        raise ValueError("Azure OpenAI response was not a JSON array or object.")

    # ── 4. Validate rules ────────────────────────────────────────────────────
    valid_rules = []
    for rule in new_rules:
        if not isinstance(rule, dict):
            continue
        if not {"raw_symptom", "node_type", "condition", "test", "source"}.issubset(rule):
            continue
        if rule["node_type"] not in VALID_NODE_TYPES:
            rule["node_type"] = "Symptom"
        rule["test"] = diagnostic_test
        valid_rules.append(rule)

    if not valid_rules:
        raise ValueError(
            f"Azure OpenAI generated {len(new_rules)} raw rules but none passed schema validation."
        )

    # ── 5. Append to guideline_rules.json ────────────────────────────────────
    with open(_RULES_FILE, "r") as f:
        existing = json.load(f)

    existing.append({"_comment": f"PATHWAY: {diagnostic_test} — LLM-generated via Diagnotix"})
    existing.extend(valid_rules)

    with open(_RULES_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    # ── 6. Rebuild graph ─────────────────────────────────────────────────────
    stats = generate_knowledge_graph()

    # ── 7. Compute diff ──────────────────────────────────────────────────────
    full_graph = load_graph_json()

    new_nodes = [n for n in full_graph["nodes"] if n["id"] not in existing_node_ids]
    new_node_ids = {n["id"] for n in new_nodes}

    # Include any edge that touches at least one new node
    new_edges = [
        e for e in full_graph["edges"]
        if e["source"] in new_node_ids or e["target"] in new_node_ids
    ]

    return AddTestResponse(
        success=True,
        diagnostic_test=diagnostic_test,
        new_rules_added=len(valid_rules),
        new_nodes=new_nodes,
        new_edges=new_edges,
        total_nodes=stats["nodes"],
        total_edges=stats["edges"],
        message=(
            f"Added {len(valid_rules)} rules for '{diagnostic_test}'. "
            f"Graph now has {stats['nodes']} nodes and {stats['edges']} edges."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Async entry point
# ─────────────────────────────────────────────────────────────────────────────

async def add_test(diagnostic_test: str) -> AddTestResponse:
    """Run the synchronous pipeline in a thread pool so the FastAPI event
    loop is not blocked during the multi-minute API call chain."""
    return await asyncio.to_thread(_sync_add_test, diagnostic_test)
