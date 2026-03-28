"""kg_service.py
================
Orchestrates the Diagnotix add-test pipeline:
  1. Snapshot the current graph node IDs (for diff computation).
  2. Call the configured LLM to extract triage rules for the requested diagnostic test.
     Supported providers (set LLM_PROVIDER in .env):
       - "nebius"  → Nebius AI Studio (OpenAI-compatible, API key auth)
       - "azure"   → Azure OpenAI     (AzureCliCredential, requires az login)
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
from audit_guidelines import _normalise_rule, run_grounded_verify_pass  # noqa: E402

_RULES_FILE = os.path.join(_KG_DIR, "guideline_rules.json")

# ─────────────────────────────────────────────────────────────────────────────
# LLM client — controlled by LLM_PROVIDER env var ("azure" or "nebius").
#
# Nebius:  set LLM_PROVIDER=nebius, NEBIUS_API_KEY, and optionally NEBIUS_MODEL.
#          Uses the standard openai.OpenAI client against Nebius AI Studio's
#          OpenAI-compatible endpoint.
#
# Azure:   set LLM_PROVIDER=azure (default), ENDPOINT_URL, DEPLOYMENT_NAME.
#          Uses AzureOpenAI with AzureCliCredential (requires az login).
#
# Optionally set BING_SEARCH_KEY to enable live web search grounding for
# guidelines outside the four encoded sources.  Without it the LLM falls back
# to training knowledge.
# ─────────────────────────────────────────────────────────────────────────────
_bing_key = os.environ.get("BING_SEARCH_KEY")
_provider = os.environ.get("LLM_PROVIDER", "azure").lower()

if _provider == "nebius":
    from openai import OpenAI as _OpenAI
    _nebius_key = os.environ.get("NEBIUS_API_KEY")
    if not _nebius_key:
        raise ValueError(
            "NEBIUS_API_KEY not set — required when LLM_PROVIDER=nebius. "
            "Add it to knowledge-graphs/.env and restart the server."
        )
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
    _endpoint = os.environ.get("ENDPOINT_URL")
    _deployment = os.environ.get("DEPLOYMENT_NAME")
    if not _endpoint:
        raise ValueError(
            "ENDPOINT_URL not set — required when LLM_PROVIDER=azure. "
            "Add it to knowledge-graphs/.env and restart the server."
        )
    if not _deployment:
        raise ValueError(
            "DEPLOYMENT_NAME not set — required when LLM_PROVIDER=azure. "
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

# Recognised authoritative guideline bodies — rules citing anything outside
# this set are rejected at the schema-validation stage before the LLM review.
# Detects LLM-generated condition names with a spurious short prefix that duplicates
# the start of the next word (e.g. "Ac Acute Myocardial Infarction", "St Stroke").
# Pattern: standalone word of 1–3 chars followed by a word beginning with those chars.
_GARBLED_COND_RE = re.compile(r"^(\w{1,3})\s+\1", re.IGNORECASE)

VALID_SOURCES = {
    "AHA/ACC", "CTAS", "ACR", "NICE", "ESC", "ACEP", "WHO", "IDSA", "GOLD",
    "ACOG", "NCCN", "Endocrine Society", "ASA", "AASLD", "ADA", "ATS", "ERS",
    "ISHLT", "BSH", "SIGN", "EASL", "EULAR", "ACG", "AGA", "SAGES", "STS",
    "ACC", "AHA", "CHEST", "USPSTF", "CDC", "AAP", "AAFP", "CCS", "CTS",
}

VERIFICATION_PROMPT = """\
You are a senior clinical knowledge reviewer auditing LLM-generated triage rules before
they are committed to a medical knowledge graph used in real clinical decision support.
Your only job is to REMOVE inaccurate or unsupported rules — do not add or rewrite.

For each rule in the array you receive, verify ALL three criteria:

1. SOURCE accuracy — does the cited guideline body (the "source" field) actually publish
   formal recommendations for the given diagnostic test? A source is invalid if it is
   fabricated, misattributed, or merely tangentially related.

2. CLINICAL plausibility — is the raw_symptom → condition → test pathway an explicitly
   recognised clinical indication in the cited guideline? Generic pairings that are not
   directly backed by a specific recommendation must be removed.

3. CONDITION specificity — the "condition" must be a real, named diagnosis (e.g.
   "Pulmonary Embolism", not "Respiratory Distress"). Vague descriptors are not conditions.
   NOTE: ICD-10 codes containing "unspecified" refer to coding granularity, not vagueness —
   do NOT remove a rule solely because the condition string contains "unspecified".
   NOTE: ICD-10 codes containing "sequela" denote the same underlying diagnosis expressed
   as a late-effect encounter code — the clinical pathway remains valid. Do NOT remove a
   rule solely because the condition string contains "sequela".

Return ONLY the rules that pass all three checks, unchanged, as a plain JSON array.
Do NOT add new rules. When in doubt, remove — a false negative is safer than a false positive.
Output valid JSON only (a bare array, no markdown fences, no explanation).\
"""

SYSTEM_PROMPT = """\
You are a clinical knowledge engineer that populates a medical triage knowledge graph.
Given a diagnostic test name, generate triage rules grounded in AUTHORITATIVE CLINICAL
GUIDELINES only.

GUIDELINE GROUNDING REQUIREMENTS (critical):
- The four guidelines already encoded in the system are: AHA/ACC, CTAS, ACR, NICE.
  Prefer these when they cover the diagnostic test.
- For any test not covered by those four, you MUST identify the relevant authoritative
  source (e.g. ESC, ACEP, WHO, IDSA, GOLD, ACOG, Endocrine Society, NCCN, UpToDate)
  and base every rule on specific recommendations from that source.
- The "source" field must be a real, recognised guideline body abbreviation — never
  invent a source or use a generic label like "Clinical Practice".
- Every condition you name must be one for which the test is formally recommended by the
  cited guideline; do not include speculative or unsupported pairings.

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
  "source"       (required) — authoritative guideline abbreviation
                              (e.g. "ACR", "AHA/ACC", "NICE", "ESC", "ACEP", "WHO", "IDSA")
  "treatment"    (optional) — include ONLY when there is a well-established first-line
                              treatment explicitly recommended by the cited guideline
                              (e.g. "Aspirin", "Alteplase")

Generate 8–15 rules spanning at least 3 distinct conditions. Be clinically accurate.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Verification helper
# ─────────────────────────────────────────────────────────────────────────────

def _verify_rules(rules: list[dict], diagnostic_test: str) -> list[dict]:
    """Send generated rules to the LLM for a second-pass clinical accuracy review.

    The model acts as a clinical reviewer and returns only the rules that pass
    three checks: source accuracy, clinical plausibility, and condition specificity.
    If the verification call itself fails, the original rules are returned unchanged
    so the pipeline is not blocked.
    """
    if not rules:
        return rules

    try:
        response = _CLIENT.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": VERIFICATION_PROMPT},
                {"role": "user", "content": (
                    f'Diagnostic test under review: "{diagnostic_test}"\n\n'
                    f"Rules to verify:\n{json.dumps(rules, indent=2)}"
                )},
            ],
        )
        text = response.choices[0].message.content or ""
        try:
            verified = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            verified = json.loads(match.group()) if match else None

        if isinstance(verified, dict):
            verified = next((v for v in verified.values() if isinstance(v, list)), None)

        if not isinstance(verified, list):
            print("  [verify] Could not parse verification response — keeping original rules.")
            return rules

        dropped = len(rules) - len(verified)
        if dropped > 0:
            print(f"  [verify] Removed {dropped} rule(s) that failed clinical review.")
        else:
            print(f"  [verify] All {len(rules)} rule(s) passed clinical review.")
        return verified

    except Exception as e:
        print(f"  [verify] Verification call failed ({e}) — keeping original rules.")
        return rules


# ─────────────────────────────────────────────────────────────────────────────
# Targeted regeneration helper
# ─────────────────────────────────────────────────────────────────────────────

def _regenerate_targeted(
    dropped_pairs: list[dict],
    diagnostic_test: str,
) -> list[dict]:
    """Ask the LLM to regenerate rules specifically for (condition, node_type) pairs
    that were removed during a previous verification pass.

    Args:
        dropped_pairs: List of dicts with "condition" and "node_type" keys from
                       the rules that failed verification.
        diagnostic_test: The diagnostic test string (used for the "test" field).

    Returns:
        Schema-validated replacement rules (may be empty if generation fails).
    """
    if not dropped_pairs:
        return []

    pair_lines = "\n".join(
        f"  - condition: \"{p['condition']}\", node_type: \"{p['node_type']}\""
        for p in dropped_pairs
    )
    user_message = (
        f'The following (condition, node_type) pairs were removed during clinical '
        f'review for the diagnostic test "{diagnostic_test}". Generate replacement '
        f'rules that are more precisely grounded in authoritative guidelines.\n\n'
        f'Pairs to replace:\n{pair_lines}\n\n'
        f'IMPORTANT: the "test" field must be exactly "{diagnostic_test}" for every rule.\n'
        f'Generate one rule per pair listed above. Return a JSON array only.'
    )
    try:
        response = _CLIENT.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            parsed = json.loads(match.group()) if match else []

        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
        if not isinstance(parsed, list):
            return []

        # Schema validation
        REQUIRED = {"raw_symptom", "node_type", "condition", "test", "source"}
        valid: list[dict] = []
        for rule in parsed:
            if not isinstance(rule, dict):
                continue
            if not REQUIRED.issubset(rule):
                continue
            if rule["node_type"] not in VALID_NODE_TYPES:
                rule["node_type"] = "Symptom"
            if rule["source"] not in VALID_SOURCES:
                continue
            if _GARBLED_COND_RE.match(rule.get("condition", "")):
                print(f"  [regen] Dropped garbled condition: '{rule['condition']}'")
                continue
            rule["test"] = diagnostic_test
            valid.append(rule)

        print(f"  [regen] {len(valid)} replacement rule(s) generated for {len(dropped_pairs)} dropped pair(s).")
        return valid

    except Exception as e:
        print(f"  [regen] Targeted regeneration failed ({e}) — skipping.")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Iterative normalise → verify → regenerate loop
# ─────────────────────────────────────────────────────────────────────────────

def _verify_and_converge(
    initial_rules: list[dict],
    diagnostic_test: str,
    max_iterations: int = 3,
) -> list[dict]:
    """Iteratively normalise, verify, and regenerate rules until stable.

    Each iteration:
      1. Normalises condition/symptom names using ICD-10-CM and HP/MONDO ontologies
         (via _normalise_rule from audit_guidelines) so the LLM reviewer sees clean
         clinical terms rather than raw LLM-invented strings.
      2. Verifies the normalised rules with the LLM clinical reviewer.
      3. If rules were dropped and iterations remain, calls targeted regeneration
         for the dropped (condition, node_type) pairs and adds candidates back into
         the pool for the next iteration.

    Converges when 0 rules are dropped in a pass, or max_iterations is reached.

    Args:
        initial_rules:  Schema-valid rules from the initial LLM generation pass.
        diagnostic_test: The diagnostic test name.
        max_iterations:  Maximum normalise→verify→regen cycles (default 3).

    Returns:
        The surviving rules after convergence.
    """
    current_rules = list(initial_rules)

    for iteration in range(1, max_iterations + 1):
        print(f"  [loop] Iteration {iteration}/{max_iterations}: "
              f"{len(current_rules)} candidate rule(s).")

        # Steps 1-2: Normalise + gather PMC/web evidence + grounded LLM verify
        survived = run_grounded_verify_pass(current_rules, diagnostic_test)
        dropped_count = len(current_rules) - len(survived)

        if dropped_count == 0:
            print(f"  [loop] Converged after {iteration} iteration(s) — "
                  f"{len(survived)} rule(s) stable.")
            return survived

        print(f"  [loop] {dropped_count} rule(s) dropped in iteration {iteration}.")

        # Step 3: Targeted regeneration for dropped pairs (if iterations remain)
        if iteration < max_iterations:
            norm_survived_keys = {
                (r["condition"].strip().lower(), r["node_type"].strip().lower())
                for r in survived
            }
            # Normalise current_rules so condition strings match what run_grounded_verify_pass
            # returned in `survived` — without this, survived rules whose condition was
            # normalised (e.g. "Ectopic Pregnancy" → canonical ICD-10 name) won't be found
            # in norm_survived_keys, causing them to appear as "dropped" and be re-generated
            # unnecessarily.  _normalise_rule uses module-level caches so no extra API calls.
            norm_current = [_normalise_rule(r)[0] for r in current_rules]
            dropped_pairs = [
                {"condition": r["condition"], "node_type": r["node_type"]}
                for r in norm_current
                if (r["condition"].strip().lower(), r["node_type"].strip().lower())
                not in norm_survived_keys
            ]
            replacements = _regenerate_targeted(dropped_pairs, diagnostic_test)
            current_rules = survived + replacements
        else:
            current_rules = survived

    print(f"  [loop] Reached max_iterations ({max_iterations}) — "
          f"returning {len(current_rules)} surviving rule(s).")
    return current_rules


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous worker (runs in a thread-pool via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────────

def _sync_add_test(diagnostic_test: str) -> AddTestResponse:
    # ── 1. Snapshot existing graph node IDs before rebuild ───────────────────
    existing_node_ids = get_existing_node_ids()

    # ── 2. LLM extraction ────────────────────────────────────────────────────
    user_message = (
        f'Generate triage rules for the diagnostic test: "{diagnostic_test}"\n\n'
        f"The system already encodes rules from AHA/ACC, CTAS, ACR, and NICE. "
        f"If '{diagnostic_test}' is covered by one of those four guidelines, use them. "
        f"If it requires a different guideline body (e.g. ESC, ACEP, IDSA, GOLD, ACOG), "
        f"identify the correct authoritative source and base every rule on its specific "
        f"recommendations.\n\n"
        f"Cover at least 3–4 different conditions this test can diagnose and include "
        f"a mix of node types (Symptom, Vital_Sign_Threshold, Risk_Factor, etc.).\n\n"
        f'IMPORTANT: the "test" field must be exactly "{diagnostic_test}" for every rule.'
    )

    # Attach Bing Search grounding when a key is configured so the model can
    # retrieve up-to-date guideline content for tests outside the four encoded
    # sources.  Falls back to training knowledge when the key is absent.
    extra_kwargs: dict = {}
    if _bing_key:
        extra_kwargs["extra_body"] = {
            "data_sources": [
                {
                    "type": "bing_search",
                    "parameters": {
                        "endpoint": "https://api.bing.microsoft.com/",
                        "key": _bing_key,
                        "search_strictness": 3,
                    },
                }
            ]
        }

    response = _CLIENT.chat.completions.create(
        model=_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # response_format={"type": "json_object"}, # commented out because the older model version
        **extra_kwargs,
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

    # ── 4. Schema validation ─────────────────────────────────────────────────
    REQUIRED = {"raw_symptom", "node_type", "condition", "test", "source"}
    valid_rules = []
    for rule in new_rules:
        if not isinstance(rule, dict):
            continue
        if not REQUIRED.issubset(rule):
            continue
        if rule["node_type"] not in VALID_NODE_TYPES:
            rule["node_type"] = "Symptom"
        if rule["source"] not in VALID_SOURCES:
            print(f"  [schema] Dropped rule with unrecognised source '{rule['source']}'.")
            continue
        rule["test"] = diagnostic_test
        valid_rules.append(rule)

    if not valid_rules:
        raise ValueError(
            f"Azure OpenAI generated {len(new_rules)} raw rules but none passed schema validation."
        )

    # ── 4b. Iterative normalise → verify → regenerate loop ───────────────────
    print(f"  Starting verify-and-converge loop on {len(valid_rules)} schema-valid rule(s)...")
    valid_rules = _verify_and_converge(valid_rules, diagnostic_test)

    if not valid_rules:
        raise ValueError(
            "All generated rules were removed during clinical verification. "
            "Try rephrasing the test name or check that it has authoritative guideline coverage."
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
