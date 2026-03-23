"""audit_guidelines.py
======================
Retroactively audits guideline_rules.json in three passes (run in order):

  Pass 1 — Name normalisation (always runs first)
    Replaces LLM-invented condition and symptom strings with canonical names
    from authoritative ontologies before any LLM review or evidence queries:
      • condition   → ICD-10-CM canonical description (NLM Clinical Tables API)
      • raw_symptom → HP / MONDO canonical label (EMBL-EBI OLS4 API)
    The original string is kept when no high-confidence ontology match is found.
    Running this first ensures the LLM sees clean clinical names, not verbose
    ICD-10 boilerplate strings like "…, unspecified" or "…, sequela".

  Pass 2 — Evidence Grounding (optional, --grounding flag)
    Queries Europe PMC co-occurrence and DuckDuckGo for each rule and annotates
    it with a PMC evidence tier.  Rules with zero evidence on both signals can
    optionally be written to flagged_rules.json (--flag-low-evidence).

  Pass 3 — LLM Verification
    Groups rules by diagnostic test (pathway) and sends each group through
    the same LLM clinical reviewer used by kg_service.py.  Rules that fail
    (fabricated source, implausible pairing, vague condition) are removed.
    When grounding is enabled, the evidence tiers are injected into the prompt
    so the LLM can weight its decision accordingly.

Usage
-----
  python audit_guidelines.py                                    # normalise + verify, write in-place
  python audit_guidelines.py --dry-run                         # report only, no file changes
  python audit_guidelines.py --skip-verify                     # normalise names only
  python audit_guidelines.py --skip-normalise                  # verify only
  python audit_guidelines.py --rules-json /path/to/guideline_rules.json
  python audit_guidelines.py --grounding                       # Pass 2: PMC + web grounding before LLM review
  python audit_guidelines.py --grounding --flag-low-evidence   # also writes flagged_rules.json
  python audit_guidelines.py --grounding --dry-run             # grounding report only
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from azure.identity import AzureCliCredential, get_bearer_token_provider
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))

UMLS_API_KEY = os.getenv("UMLS_API_KEY")

_endpoint   = os.environ.get("ENDPOINT_URL")
_deployment = os.environ.get("DEPLOYMENT_NAME")

if not _endpoint or not _deployment:
    raise ValueError("ENDPOINT_URL / DEPLOYMENT_NAME missing from knowledge-graphs/.env")

_credential     = AzureCliCredential()
_token_provider = get_bearer_token_provider(
    _credential, "https://cognitiveservices.azure.com/.default"
)
_CLIENT = AzureOpenAI(
    azure_endpoint=_endpoint,
    azure_ad_token_provider=_token_provider,
    api_version="2025-01-01-preview",
)

# ─────────────────────────────────────────────────────────────────────────────
# Verification (same prompt as kg_service.py)
# ─────────────────────────────────────────────────────────────────────────────

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
   NOTE: ICD-10 codes that contain the word "unspecified" (e.g. "Unspecified acute appendicitis",
   "Abdominal aortic aneurysm, without rupture, unspecified") ARE real, valid named diagnoses —
   the word refers to coding granularity, not diagnostic vagueness. Do NOT remove a rule solely
   because the condition string contains "unspecified".
   NOTE: ICD-10 codes that contain the word "sequela" denote the same underlying diagnosis
   expressed as a late-effect encounter code — the clinical pathway remains valid. Do NOT
   remove a rule solely because the condition string contains "sequela".

Return ONLY the rules that pass all three checks, unchanged, as a plain JSON array.
Do NOT add new rules. When in doubt and evidence is absent, remove. When evidence is present,
retain unless the rule clearly fails criterion 1 or 2.
Output valid JSON only (a bare array, no markdown fences, no explanation).\
"""

# Maximum rules sent to the LLM in a single verification call.
# Large batches (>20) cause the model to drop everything indiscriminately.
_MAX_VERIFY_BATCH = 15

VALID_SOURCES = {
    "AHA/ACC", "CTAS", "ACR", "NICE", "ESC", "ACEP", "WHO", "IDSA", "GOLD",
    "ACOG", "NCCN", "Endocrine Society", "ASA", "AASLD", "ADA", "ATS", "ERS",
    "ISHLT", "BSH", "SIGN", "EASL", "EULAR", "ACG", "AGA", "SAGES", "STS",
    "ACC", "AHA", "CHEST", "USPSTF", "CDC", "AAP", "AAFP", "CCS", "CTS",
}


def _verify_group(
    rules: list[dict],
    test_name: str,
    evidence_map: dict[int, dict] | None = None,
) -> list[dict]:
    """Send one pathway's rules to the LLM for clinical review.  Returns the surviving subset.

    Args:
        rules:        Rules for a single diagnostic test pathway.
        test_name:    The diagnostic test name (used in the prompt).
        evidence_map: Optional dict keyed by rule index (0-based within `rules`).
                      Each value is the dict returned by _gather_evidence().
                      When provided, PMC counts and web snippets are appended to
                      the user message and a guidance prefix is added to the system
                      prompt so the LLM can use the evidence in its review.
    """
    # Build optional evidence block to append to the user message
    evidence_block = ""
    system_prefix  = ""
    if evidence_map:
        lines = ["\n\n[EVIDENCE — use to inform criteria 1 and 2]"]
        for i, rule in enumerate(rules):
            ev = evidence_map.get(i)
            if ev is None:
                continue
            pmc_n = ev["pmc_articles"]
            if pmc_n >= 5000:
                tier = f"STRONG ({pmc_n:,} PMC articles — retain unless clearly wrong)"
            elif pmc_n >= 500:
                tier = f"GOOD ({pmc_n:,} PMC articles — prefer to retain)"
            elif pmc_n >= 50:
                tier = f"MODERATE ({pmc_n} PMC articles)"
            elif pmc_n > 0:
                tier = f"LIMITED ({pmc_n} PMC articles)"
            else:
                tier = "NONE (0 PMC articles — apply extra scrutiny)"
            label = f"Rule {i + 1} ({rule['condition']} → {rule['test']})"
            lines.append(f"\n{label}:")
            lines.append(f"  PMC evidence tier: {tier}")
            lines.append(f"  PMC patents: {ev['pmc_patents']}")
            lines.append(f"  Web search query: \"{ev['ddg_query']}\"")
            if ev["ddg_snippets"]:
                lines.append(f"  Web snippets ({len(ev['ddg_snippets'])} found):")
                for snippet in ev["ddg_snippets"]:
                    lines.append(f"    - {snippet}")
            else:
                lines.append("  Web snippets: none found")
        lines.append("\n[/EVIDENCE]")
        evidence_block = "\n".join(lines)

        system_prefix = (
            "An [EVIDENCE] block follows the rules. Each rule carries a PMC evidence tier.\n"
            "The tier OVERRIDES the default 'when in doubt, remove' instruction:\n"
            "  • STRONG / GOOD tier: RETAIN by default. Only remove if the source field is "
            "clearly fabricated (not a real guideline body) or the condition→test pairing is "
            "clinically absurd. Substantial PMC literature is proof enough of clinical validity.\n"
            "  • MODERATE tier: evaluate on clinical merits; lean toward retaining.\n"
            "  • LIMITED tier: apply all three criteria normally.\n"
            "  • NONE tier (0 PMC articles, no web snippets): apply extra scrutiny; "
            "bias toward removal when source or plausibility is uncertain.\n"
            "Absence of web snippets alone is NOT grounds for removal when PMC counts are "
            "substantial — web search coverage of clinical guidelines is incomplete.\n\n"
        )

    try:
        response = _CLIENT.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": system_prefix + VERIFICATION_PROMPT},
                {"role": "user", "content": (
                    f'Diagnostic test under review: "{test_name}"\n\n'
                    f"Rules to verify:\n{json.dumps(rules, indent=2)}"
                    + evidence_block
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
            print(f"    [verify] Could not parse response for '{test_name}' — keeping originals.")
            return rules

        dropped = len(rules) - len(verified)
        print(f"    [verify] '{test_name}': {len(verified)} kept, {dropped} removed.")
        return verified

    except Exception as e:
        print(f"    [verify] LLM call failed for '{test_name}': {e} — keeping originals.")
        return rules


# ─────────────────────────────────────────────────────────────────────────────
# Name normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_VALID_OLS_PREFIXES = ("HP_", "MONDO_", "EFO_")
_nlm_cache:  dict = {}
_ols_cache:  dict = {}

# ── Grounding caches & checkpoint ─────────────────────────────────────────────
_pmc_grounding_cache: dict = {}   # (condition.lower(), test.lower()) → {"articles": int, "patents": int}
_ddg_grounding_cache: dict = {}   # query_str.lower() → list[str]
_GROUNDING_CHECKPOINT_PATH = os.path.join(_THIS_DIR, ".audit_grounding_cache.json")


def _load_grounding_checkpoint() -> None:
    """Populate in-memory grounding caches from the sidecar checkpoint file."""
    if not os.path.exists(_GROUNDING_CHECKPOINT_PATH):
        return
    try:
        with open(_GROUNDING_CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if key.startswith("pmc:"):
                parts = key[4:].split("|", 1)
                if len(parts) == 2:
                    _pmc_grounding_cache[(parts[0], parts[1])] = value
            elif key.startswith("ddg:"):
                _ddg_grounding_cache[key[4:]] = value
        print(f"  [grounding] Resumed from checkpoint ({len(data)} cached entries).")
    except Exception:
        pass  # corrupt checkpoint — start fresh


def _save_grounding_checkpoint() -> None:
    """Persist grounding caches to the sidecar checkpoint file."""
    try:
        data = {}
        for (condition, test), value in _pmc_grounding_cache.items():
            data[f"pmc:{condition}|{test}"] = value
        for query, snippets in _ddg_grounding_cache.items():
            data[f"ddg:{query}"] = snippets
        with open(_GROUNDING_CHECKPOINT_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # never crash the audit over a checkpoint write failure


def _pmc_cooccurrence(condition: str, test: str) -> dict:
    """Return Europe PMC co-occurrence counts for a (condition, test) pair.

    Returns {"articles": int, "patents": int}.  Uses the same retry/backoff
    pattern as build_kg.get_literature_breakdown — reimplemented here to avoid
    importing build_kg.py (which runs top-level setup code on import).
    """
    key = (condition.lower(), test.lower())
    if key in _pmc_grounding_cache:
        return _pmc_grounding_cache[key]

    base_url   = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    base_query = f"({condition}) AND ({test})"

    def _count(extra_filter: str, retries: int = 3, backoff: float = 2.0) -> int:
        for attempt in range(retries):
            try:
                r = requests.get(
                    base_url,
                    params={"query": f"{base_query} {extra_filter}",
                            "format": "json", "pageSize": 1},
                    timeout=15,
                )
                r.raise_for_status()
                return r.json().get("hitCount", 0)
            except requests.exceptions.HTTPError:
                if r.status_code in (503, 429, 502) and attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    print(f"    [pmc] {r.status_code} for '{condition}' + '{test}' — retry in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    return 0
            except Exception:
                return 0
        return 0

    result = {"articles": _count("NOT SRC:PAT"), "patents": _count("SRC:PAT")}
    _pmc_grounding_cache[key] = result
    _save_grounding_checkpoint()
    return result


def _ddg_web_search(query: str) -> list[str]:
    """Query DuckDuckGo (no API key required) and return up to 5 result snippets."""
    key = query.lower().strip()
    if key in _ddg_grounding_cache:
        return _ddg_grounding_cache[key]

    snippets: list[str] = []
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; audit_guidelines/1.0)"},
            timeout=10,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            elements = soup.select(".result__snippet")
            if elements:
                snippets = [el.get_text(strip=True) for el in elements[:5]]
            else:
                # Fallback: use link text if snippets selector missed
                snippets = [
                    a.get_text(strip=True)
                    for a in soup.select(".result__a")[:5]
                ]
        time.sleep(1.0)  # avoid rate-limiting
    except Exception as exc:
        print(f"    [ddg] Search failed for query '{query[:60]}...': {exc}")

    _ddg_grounding_cache[key] = snippets
    _save_grounding_checkpoint()
    return snippets


_ICD10_JARGON_RE = re.compile(
    r"\b(unspecified|NOS|sequela|initial encounter|subsequent encounter|"
    r"not elsewhere classified|without intrauterine pregnancy|"
    r"without mention of|type [0-9]+)\b",
    re.IGNORECASE,
)

# Encounter-type modifiers that make ICD-10 codes inappropriate for acute triage
# rules.  Used by _icd10_canonical_name() to skip bad normalisation candidates.
_ICD10_AVOID_RE = re.compile(
    r"\b(sequela|subsequent encounter|initial encounter|late effect)\b",
    re.IGNORECASE,
)


def _strip_icd10_jargon(condition: str) -> str:
    """Remove ICD-10 boilerplate words that hurt web-search quality.

    E.g. "Acute myocardial infarction, unspecified" → "Acute myocardial infarction"
         "Unspecified fracture of shaft of humerus, right arm, sequela" → "fracture of shaft of humerus, right arm"
    """
    bare = _ICD10_JARGON_RE.sub("", condition)
    bare = re.sub(r"[\s,;]+", " ", bare).strip().strip(",").strip()
    return bare if bare else condition  # fallback to original if everything was stripped


def _gather_evidence(rule: dict) -> dict:
    """Fetch PMC co-occurrence and DuckDuckGo snippets for a single rule.

    Returns:
        {
            "pmc_articles": int,
            "pmc_patents":  int,
            "ddg_snippets": list[str],
            "ddg_query":    str,
        }
    """
    condition = rule["condition"]
    test      = rule["test"]
    source    = rule["source"]

    pmc = _pmc_cooccurrence(condition, test)
    bare_condition = _strip_icd10_jargon(condition)
    ddg_query = f"{source} {bare_condition} {test} guideline"
    snippets  = _ddg_web_search(ddg_query)

    return {
        "pmc_articles": pmc["articles"],
        "pmc_patents":  pmc["patents"],
        "ddg_snippets": snippets,
        "ddg_query":    ddg_query,
    }


def _is_low_evidence(evidence: dict) -> bool:
    """Return True when a rule has zero PMC articles AND zero web snippets."""
    return evidence["pmc_articles"] == 0 and len(evidence["ddg_snippets"]) == 0


def _icd10_canonical_name(term: str) -> str | None:
    """Return the UMLS canonical name for term (aligned with ICD-10 international), or None.

    Uses UMLS normalizedString search — no encounter-modifier filtering needed
    since WHO ICD-10 does not use ICD-10-CM-specific modifiers like 'sequela'
    or 'initial encounter'. Keeps the rupture-mismatch guard to avoid candidates
    that incorrectly introduce 'ruptured' when the original term does not contain it.
    Falls back to the first result only if every candidate fails the filter.
    """
    key = term.lower()
    if key in _nlm_cache:
        return _nlm_cache[key]
    if not UMLS_API_KEY:
        _nlm_cache[key] = None
        return None
    term_has_rupture = bool(re.search(r"\brupt", term, re.IGNORECASE))
    try:
        r = requests.get(
            "https://uts-ws.nlm.nih.gov/rest/search/current",
            params={"string": term, "searchType": "normalizedString", "apiKey": UMLS_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("result", {}).get("results", [])
        name = None
        for res in results:
            candidate = res.get("name", "")
            if not term_has_rupture and re.search(r"\brupt", candidate, re.IGNORECASE):
                continue
            name = candidate
            break
        if name is None and results:
            name = results[0]["name"]
    except Exception:
        name = None
    _nlm_cache[key] = name
    return name


def _hp_mondo_canonical_label(term: str) -> str | None:
    """Return the HP/MONDO canonical label for term, or None.

    A candidate label is only accepted when it meets two quality checks:
      1. At least 2 words — rejects single-word generic matches like "right".
      2. At least one meaningful word (> 3 chars) from the original term appears
         in the candidate — ensures topical relevance.
    """
    key = term.lower()
    if key in _ols_cache:
        return _ols_cache[key]

    # Meaningful words from the original term (length > 3, lower-cased)
    original_words = {w for w in re.split(r"\W+", term.lower()) if len(w) > 3}

    try:
        r = requests.get(
            "https://www.ebi.ac.uk/ols4/api/search",
            params={"q": term, "ontology": "hp,mondo,efo",
                    "queryFields": "label,synonym", "exact": "false", "rows": 5},
            timeout=10,
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        label = None
        for doc in docs:
            if not doc.get("short_form", "").startswith(_VALID_OLS_PREFIXES):
                continue
            candidate = doc.get("label", "")
            candidate_words = set(re.split(r"\W+", candidate.lower()))
            # Reject single-word or non-overlapping candidates
            if len(candidate.split()) < 2:
                continue
            if original_words and not original_words.intersection(candidate_words):
                continue
            label = candidate
            break
    except Exception:
        label = None
    _ols_cache[key] = label
    return label


def _normalise_rule(rule: dict) -> tuple[dict, list[str]]:
    """Return (possibly updated rule, list of change descriptions)."""
    changes: list[str] = []
    rule = dict(rule)

    # Normalise condition → ICD-10-CM canonical description
    cond = rule.get("condition", "")
    canonical_cond = _icd10_canonical_name(cond)
    if canonical_cond and canonical_cond.lower() != cond.lower():
        changes.append(f"condition: '{cond}' → '{canonical_cond}'")
        rule["condition"] = canonical_cond

    # Normalise raw_symptom → HP/MONDO canonical label
    sym = rule.get("raw_symptom", "")
    canonical_sym = _hp_mondo_canonical_label(sym)
    if canonical_sym and canonical_sym.lower() != sym.lower():
        changes.append(f"raw_symptom: '{sym}' → '{canonical_sym}'")
        rule["raw_symptom"] = canonical_sym

    return rule, changes


# ─────────────────────────────────────────────────────────────────────────────
# Shared grounded verification pass (used by generation pipelines)
# ─────────────────────────────────────────────────────────────────────────────

def run_grounded_verify_pass(rules: list[dict], test_name: str) -> list[dict]:
    """Run one normalise → evidence-gather → grounded-verify pass on a rule set.

    Intended to be imported by generation pipelines (kg_service.py,
    kg_enrichment_pipeline.py) so that every rule written to
    guideline_rules.json has passed the same PMC-evidence-aware clinical review
    used by the standalone audit_guidelines CLI.

    Steps:
      1. Normalise condition/symptom strings via ICD-10-CM (NLM) and HP/MONDO
         (EMBL-EBI OLS4) so the LLM reviewer sees canonical clinical names.
      2. Gather PMC co-occurrence counts and DuckDuckGo snippets per rule via
         _gather_evidence — results are cached in memory and persisted to the
         grounding checkpoint so repeated calls are cheap.
      3. Call _verify_group with the populated evidence_map so the LLM reviewer
         receives PMC tier guidance (STRONG/GOOD/MODERATE/LIMITED/NONE) before
         deciding whether to retain or drop each rule.

    Args:
        rules:     A list of schema-valid rule dicts (raw_symptom, node_type,
                   condition, test, source all present).
        test_name: The diagnostic test name — passed to the LLM reviewer prompt.

    Returns:
        The subset of rules that passed all three verification criteria.
    """
    if not rules:
        return rules

    _load_grounding_checkpoint()

    # Step 1: normalise
    normalised: list[dict] = []
    for rule in rules:
        updated, changes = _normalise_rule(rule)
        if changes:
            for c in changes:
                print(f"  [normalise] {c}")
        normalised.append(updated)

    # Step 2: gather evidence
    evidence_map: dict[int, dict] = {}
    for i, rule in enumerate(normalised):
        evidence_map[i] = _gather_evidence(rule)

    # Step 3: grounded verify
    return _verify_group(normalised, test_name, evidence_map=evidence_map)


# ─────────────────────────────────────────────────────────────────────────────
# Main audit pipeline
# ─────────────────────────────────────────────────────────────────────────────

def audit(
    rules_json_path: str,
    dry_run: bool,
    skip_verify: bool,
    skip_normalise: bool,
    grounding: bool = False,
    flag_low_evidence: bool = False,
):
    with open(rules_json_path, "r") as f:
        raw = json.load(f)

    # Separate comment/marker entries from actual rules
    comments = [e for e in raw if "_comment" in e]
    rules    = [e for e in raw if "raw_symptom" in e]

    print(f"Loaded {len(rules)} rules across {len(comments)} pathway(s) from '{rules_json_path}'.")

    # ── Schema pre-filter ─────────────────────────────────────────────────────
    schema_ok = []
    schema_dropped = 0
    for rule in rules:
        if not {"raw_symptom", "node_type", "condition", "test", "source"}.issubset(rule):
            schema_dropped += 1
            continue
        if rule["source"] not in VALID_SOURCES:
            print(f"  [schema] Dropped rule with unrecognised source '{rule['source']}'.")
            schema_dropped += 1
            continue
        schema_ok.append(rule)
    print(f"\nSchema pre-filter: {len(schema_ok)} passed, {schema_dropped} dropped.")

    # ── Deduplication ─────────────────────────────────────────────────────────
    # Remove exact (raw_symptom, condition, test, source) duplicates that
    # accumulate when Diagnotix runs multiple generation passes for the same test.
    seen_keys: set[tuple] = set()
    deduped: list[dict] = []
    for rule in schema_ok:
        key = (
            rule["raw_symptom"].strip().lower(),
            rule["condition"].strip().lower(),
            rule["test"].strip().lower(),
            rule["source"].strip().lower(),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(rule)
    n_dupes = len(schema_ok) - len(deduped)
    if n_dupes:
        print(f"Deduplication: {n_dupes} exact duplicate(s) removed, {len(deduped)} unique rules remain.")
    schema_ok = deduped
    print(f"\n")

    # ── Pass 1: Name Normalisation (first — clean names before evidence/LLM) ────
    normalised_schema = schema_ok
    if not skip_normalise:
        print("=== Pass 1: Name Normalisation ===")
        normalised_schema = []
        total_changes = 0
        for rule in schema_ok:
            updated, changes = _normalise_rule(rule)
            if changes:
                total_changes += len(changes)
                for c in changes:
                    print(f"  [normalise] {c}")
            normalised_schema.append(updated)
            time.sleep(0.05)  # gentle rate limiting on NLM / OLS4
        print(f"\nNormalisation complete: {total_changes} field(s) updated.\n")

    # ── Pass 2: Evidence Grounding ────────────────────────────────────────────
    evidence_by_rule: dict[int, dict] = {}
    flagged_rules: list[dict] = []
    rules_for_verification = normalised_schema

    if grounding:
        _load_grounding_checkpoint()
        print("=== Pass 2: Evidence Grounding ===")
        low_evidence_indices: set[int] = set()

        for i, rule in enumerate(normalised_schema):
            label = f"{rule['condition']} → {rule['test']}"
            print(f"  [{i + 1}/{len(normalised_schema)}] {label}")
            ev = _gather_evidence(rule)
            evidence_by_rule[i] = ev
            print(f"    PMC: {ev['pmc_articles']} articles, {ev['pmc_patents']} patents"
                  f" | web snippets: {len(ev['ddg_snippets'])}")
            if flag_low_evidence and _is_low_evidence(ev):
                low_evidence_indices.add(i)

        if flag_low_evidence and low_evidence_indices:
            flagged_rules = [normalised_schema[i] for i in sorted(low_evidence_indices)]
            surviving_indices = [i for i in range(len(normalised_schema)) if i not in low_evidence_indices]
            rules_for_verification = [normalised_schema[i] for i in surviving_indices]
            # Re-index evidence_by_rule to match positions in rules_for_verification
            evidence_by_rule = {
                new_i: evidence_by_rule[old_i]
                for new_i, old_i in enumerate(surviving_indices)
            }
            print(f"\nEvidence grounding complete: {len(flagged_rules)} rule(s) flagged "
                  f"(zero PMC articles and zero web snippets), "
                  f"{len(rules_for_verification)} proceeding to verification.\n")
        else:
            rules_for_verification = normalised_schema
            print(f"\nEvidence grounding complete: {len(normalised_schema)} rules annotated.\n")

    # ── Pass 3: LLM Verification grouped by test ──────────────────────────────
    verified_rules = rules_for_verification
    if not skip_verify:
        print("=== Pass 3: LLM Verification ===")
        # Canonicalise test names (case-insensitive) to merge variants like
        # "Chest X-Ray" / "Chest X-ray" into a single pathway group.
        by_test: dict[str, list[dict]] = {}
        test_display: dict[str, str] = {}  # norm_key → representative display name
        for rule in rules_for_verification:
            key = rule["test"].strip().lower()
            by_test.setdefault(key, []).append(rule)
            if key not in test_display:
                test_display[key] = rule["test"].strip()

        # Build a reverse lookup so we can find each rule's index in rules_for_verification
        rule_index: dict[int, int] = {id(r): i for i, r in enumerate(rules_for_verification)}

        verified_rules = []
        for norm_key, group in by_test.items():
            test_name = test_display[norm_key]
            # Split large pathways into sub-batches to prevent LLM saturation.
            batches = [group[i:i + _MAX_VERIFY_BATCH]
                       for i in range(0, len(group), _MAX_VERIFY_BATCH)]
            batch_label = (f" (split into {len(batches)} batches of ≤{_MAX_VERIFY_BATCH})"
                           if len(batches) > 1 else "")
            print(f"  Verifying pathway '{test_name}' ({len(group)} rules{batch_label})...")

            for batch_i, batch in enumerate(batches):
                # Build a local evidence map keyed by position within this batch
                batch_evidence: dict[int, dict] | None = None
                if grounding and evidence_by_rule:
                    batch_evidence = {}
                    for local_i, rule in enumerate(batch):
                        global_i = rule_index.get(id(rule))
                        if global_i is not None and global_i in evidence_by_rule:
                            batch_evidence[local_i] = evidence_by_rule[global_i]

                surviving = _verify_group(batch, test_name, evidence_map=batch_evidence)
                if len(batches) > 1:
                    print(f"    [batch {batch_i + 1}/{len(batches)}] "
                          f"{len(surviving)}/{len(batch)} kept.")
                verified_rules.extend(surviving)

        total_dropped = len(rules_for_verification) - len(verified_rules)
        print(f"\nVerification complete: {len(verified_rules)} rules kept, {total_dropped} removed.\n")

    # ── Write flagged_rules.json ──────────────────────────────────────────────
    flagged_json_path = os.path.join(
        os.path.dirname(os.path.abspath(rules_json_path)),
        "flagged_rules.json",
    )
    if flagged_rules:
        if dry_run:
            print(f"--dry-run: would write {len(flagged_rules)} flagged rule(s) to '{flagged_json_path}'.")
        else:
            with open(flagged_json_path, "w") as f:
                json.dump(flagged_rules, f, indent=2)
            print(f"Flagged {len(flagged_rules)} low-evidence rule(s) → '{flagged_json_path}'.")

    # ── Summary ───────────────────────────────────────────────────────────────
    original_count  = len(rules)
    surviving_count = len(verified_rules)
    print("=== Summary ===")
    print(f"  Original rules  : {original_count}")
    print(f"  Flagged         : {len(flagged_rules)}")
    print(f"  After audit     : {surviving_count}")
    print(f"  Removed         : {original_count - len(flagged_rules) - surviving_count}")

    if dry_run:
        print("\n--dry-run: no file changes written.")
        return

    # Rebuild file: preserve comment markers, re-attach surviving rules after
    # their corresponding pathway comment.
    output: list[dict] = []
    by_test_final: dict[str, list[dict]] = {}
    for rule in verified_rules:
        by_test_final.setdefault(rule["test"], []).append(rule)

    # Emit comment blocks in original order, followed by their surviving rules
    seen_tests: set[str] = set()
    for entry in raw:
        if "_comment" in entry:
            output.append(entry)
            # Find which test this comment belongs to (heuristic: first rule after it)
            idx = raw.index(entry)
            for following in raw[idx + 1:]:
                if "test" in following:
                    t = following["test"]
                    if t not in seen_tests:
                        seen_tests.add(t)
                        output.extend(by_test_final.get(t, []))
                    break

    # Any rules whose test had no comment block (shouldn't happen, but safe fallback)
    for test, group in by_test_final.items():
        if test not in seen_tests:
            output.extend(group)

    with open(rules_json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Clean up grounding checkpoint on clean completion
    if grounding and os.path.exists(_GROUNDING_CHECKPOINT_PATH):
        try:
            os.remove(_GROUNDING_CHECKPOINT_PATH)
        except Exception:
            pass

    print(f"\nWritten back to '{rules_json_path}'.")
    print("Re-run 'python build_kg.py' to rebuild the knowledge graph with the audited rules.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _DEFAULT_RULES = os.path.join(_THIS_DIR, "guideline_rules.json")

    parser = argparse.ArgumentParser(description="Audit and normalise guideline_rules.json.")
    parser.add_argument(
        "--rules-json", default=_DEFAULT_RULES,
        help=f"Path to guideline_rules.json (default: {_DEFAULT_RULES})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing the file.",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the LLM verification pass; run name normalisation only.",
    )
    parser.add_argument(
        "--skip-normalise", action="store_true",
        help="Skip name normalisation; run LLM verification only.",
    )
    parser.add_argument(
        "--grounding", action="store_true",
        help=(
            "Enable Pass 0: query Europe PMC and DuckDuckGo to gather external evidence "
            "before LLM verification. Evidence is injected into the verification prompt."
        ),
    )
    parser.add_argument(
        "--flag-low-evidence", action="store_true",
        help=(
            "Requires --grounding. Rules with 0 PMC articles AND 0 web snippets are "
            "written to flagged_rules.json (sibling of guideline_rules.json) and "
            "excluded from LLM verification."
        ),
    )
    args = parser.parse_args()

    if args.flag_low_evidence and not args.grounding:
        parser.error("--flag-low-evidence requires --grounding.")

    audit(
        rules_json_path=args.rules_json,
        dry_run=args.dry_run,
        skip_verify=args.skip_verify,
        skip_normalise=args.skip_normalise,
        grounding=args.grounding,
        flag_low_evidence=args.flag_low_evidence,
    )
