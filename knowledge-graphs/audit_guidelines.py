"""audit_guidelines.py
======================
Retroactively audits guideline_rules.json in two passes:

  Pass 1 — Verification
    Groups rules by diagnostic test (pathway) and sends each group through
    the same LLM clinical reviewer used by kg_service.py.  Rules that fail
    (fabricated source, implausible pairing, vague condition) are removed.

  Pass 2 — Name normalisation
    Replaces LLM-invented condition and symptom strings with canonical names
    from authoritative ontologies:
      • condition  → ICD-10-CM canonical description (NLM Clinical Tables API)
      • raw_symptom → HP / MONDO canonical label (EMBL-EBI OLS4 API)
    The original string is kept when no high-confidence ontology match is found.

Usage
-----
  python audit_guidelines.py                     # verify + normalise, write in-place
  python audit_guidelines.py --dry-run           # report only, no file changes
  python audit_guidelines.py --skip-verify       # normalise names only
  python audit_guidelines.py --skip-normalise    # verify only
  python audit_guidelines.py --rules-json /path/to/guideline_rules.json
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))

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

Return ONLY the rules that pass all three checks, unchanged, as a plain JSON array.
Do NOT add new rules. When in doubt, remove — a false negative is safer than a false positive.
Output valid JSON only (a bare array, no markdown fences, no explanation).\
"""

VALID_SOURCES = {
    "AHA/ACC", "CTAS", "ACR", "NICE", "ESC", "ACEP", "WHO", "IDSA", "GOLD",
    "ACOG", "NCCN", "Endocrine Society", "ASA", "AASLD", "ADA", "ATS", "ERS",
    "ISHLT", "BSH", "SIGN", "EASL", "EULAR", "ACG", "AGA", "SAGES", "STS",
    "ACC", "AHA", "CHEST", "USPSTF", "CDC", "AAP", "AAFP", "CCS", "CTS",
}


def _verify_group(rules: list[dict], test_name: str) -> list[dict]:
    """Send one pathway's rules to the LLM for clinical review.  Returns the surviving subset."""
    try:
        response = _CLIENT.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": VERIFICATION_PROMPT},
                {"role": "user", "content": (
                    f'Diagnostic test under review: "{test_name}"\n\n'
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


def _icd10_canonical_name(term: str) -> str | None:
    """Return the ICD-10-CM canonical description for term, or None."""
    key = term.lower()
    if key in _nlm_cache:
        return _nlm_cache[key]
    try:
        r = requests.get(
            "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
            params={"sf": "code,name", "terms": term, "maxList": 3},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()[3] if len(r.json()) > 3 else []
        name = items[0][1] if items else None
    except Exception:
        name = None
    _nlm_cache[key] = name
    return name


def _hp_mondo_canonical_label(term: str) -> str | None:
    """Return the HP/MONDO canonical label for term, or None."""
    key = term.lower()
    if key in _ols_cache:
        return _ols_cache[key]
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
            if doc.get("short_form", "").startswith(_VALID_OLS_PREFIXES):
                label = doc.get("label")
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
# Main audit pipeline
# ─────────────────────────────────────────────────────────────────────────────

def audit(rules_json_path: str, dry_run: bool, skip_verify: bool, skip_normalise: bool):
    with open(rules_json_path, "r") as f:
        raw = json.load(f)

    # Separate comment/marker entries from actual rules
    comments = [e for e in raw if "_comment" in e]
    rules    = [e for e in raw if "raw_symptom" in e]

    print(f"Loaded {len(rules)} rules across {len(comments)} pathway(s) from '{rules_json_path}'.")

    # ── Pass 1: Schema pre-filter ─────────────────────────────────────────────
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
    print(f"\nSchema pre-filter: {len(schema_ok)} passed, {schema_dropped} dropped.\n")

    # ── Pass 2: LLM verification grouped by test ──────────────────────────────
    verified_rules = schema_ok
    if not skip_verify:
        print("=== Pass 1: LLM Verification ===")
        by_test: dict[str, list[dict]] = {}
        for rule in schema_ok:
            by_test.setdefault(rule["test"], []).append(rule)

        verified_rules = []
        for test_name, group in by_test.items():
            print(f"  Verifying pathway '{test_name}' ({len(group)} rules)...")
            surviving = _verify_group(group, test_name)
            verified_rules.extend(surviving)

        total_dropped = len(schema_ok) - len(verified_rules)
        print(f"\nVerification complete: {len(verified_rules)} rules kept, {total_dropped} removed.\n")

    # ── Pass 3: Name normalisation ────────────────────────────────────────────
    normalised_rules = verified_rules
    if not skip_normalise:
        print("=== Pass 2: Name Normalisation ===")
        normalised_rules = []
        total_changes = 0
        for rule in verified_rules:
            updated, changes = _normalise_rule(rule)
            if changes:
                total_changes += len(changes)
                for c in changes:
                    print(f"  [normalise] {c}")
            normalised_rules.append(updated)
            time.sleep(0.05)  # gentle rate limiting on NLM / OLS4
        print(f"\nNormalisation complete: {total_changes} field(s) updated.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    original_count  = len(rules)
    surviving_count = len(normalised_rules)
    print("=== Summary ===")
    print(f"  Original rules  : {original_count}")
    print(f"  After audit     : {surviving_count}")
    print(f"  Removed         : {original_count - surviving_count}")

    if dry_run:
        print("\n--dry-run: no file changes written.")
        return

    # Rebuild file: preserve comment markers, re-attach surviving rules after
    # their corresponding pathway comment.
    output: list[dict] = []
    by_test_final: dict[str, list[dict]] = {}
    for rule in normalised_rules:
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
    args = parser.parse_args()

    audit(
        rules_json_path=args.rules_json,
        dry_run=args.dry_run,
        skip_verify=args.skip_verify,
        skip_normalise=args.skip_normalise,
    )
