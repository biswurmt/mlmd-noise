"""kg_enrichment_pipeline.py
===========================
End-to-end pipeline that:

  1. Reads  data-processing/diagnosis_to_tests_graph_data.json
            (list of {"diagnosis": ..., "diagnostic_tests": [...]})
  2. Extracts unique diagnostic test names from that file.
  3. For each test not already in guideline_rules.json, calls Azure OpenAI
     to generate triage rules and appends them to guideline_rules.json.
  4. Rebuilds the knowledge graph PKL via build_kg.generate_knowledge_graph()
     (full multi-ontology API enrichment).
  5. Semantically maps the 'dx' column of an input CSV against Condition nodes
     in the rebuilt graph using Azure OpenAI, then writes an enriched CSV with
     two new columns:
       matched_graph_condition  — best matching condition node (label only)
       potential_tests          — comma-separated tests linked via REQUIRES_TEST

Usage
-----
  python kg_enrichment_pipeline.py \\
      --input-csv  /path/to/student_data.csv \\
      --output-csv patient_diagnoses_enriched.csv \\
      [--graph-json  path/to/diagnosis_to_tests_graph_data.json] \\
      [--rules-json  path/to/guideline_rules.json] \\
      [--graph-pkl   path/to/triage_knowledge_graph.pkl] \\
      [--skip-rebuild]          # skip step 3-4 if graph is already up to date

Defaults for --graph-json / --rules-json / --graph-pkl resolve relative to the
sibling knowledge-graphs/ directory so the script works from any cwd.
"""

import argparse
import json
import os
import pickle
import re
import sys

import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KG_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "knowledge-graphs"))

# Add knowledge-graphs/ to sys.path so build_kg can be imported directly.
if _KG_DIR not in sys.path:
    sys.path.insert(0, _KG_DIR)

from build_kg import generate_knowledge_graph  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Environment — try data-processing/.env first, fall back to knowledge-graphs/.env
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(_THIS_DIR, ".env"))
load_dotenv(os.path.join(_KG_DIR, ".env"))  # fallback / additional vars

_endpoint = os.environ.get("ENDPOINT_URL")
_deployment = os.environ.get("DEPLOYMENT_NAME")

if not _endpoint:
    raise ValueError("ENDPOINT_URL not found. Add it to data-processing/.env or knowledge-graphs/.env.")
if not _deployment:
    raise ValueError("DEPLOYMENT_NAME not found. Add it to data-processing/.env or knowledge-graphs/.env.")

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

_RULE_GEN_SYSTEM_PROMPT = """\
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
# Step 1 — Extract unique tests from diagnosis_to_tests_graph_data.json
# ─────────────────────────────────────────────────────────────────────────────

def extract_unique_tests(graph_json_path: str) -> list[str]:
    """Return a deduplicated list of diagnostic test names from the JSON file."""
    with open(graph_json_path, "r") as f:
        entries = json.load(f)

    seen: set[str] = set()
    tests: list[str] = []
    for entry in entries:
        for t in entry.get("diagnostic_tests", []):
            t_clean = t.strip()
            if t_clean and t_clean not in seen:
                seen.add(t_clean)
                tests.append(t_clean)

    return tests


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Generate rules for a single test via Azure OpenAI
# ─────────────────────────────────────────────────────────────────────────────

def _generate_rules_for_test(diagnostic_test: str) -> list[dict]:
    """Call Azure OpenAI and return a validated list of rule dicts for *diagnostic_test*."""
    user_message = (
        f'Generate triage rules for the diagnostic test: "{diagnostic_test}"\n\n'
        f"Cover at least 3–4 different conditions this test can diagnose and include "
        f"a mix of node types (Symptom, Vital_Sign_Threshold, Risk_Factor, etc.).\n\n"
        f'IMPORTANT: the "test" field must be exactly "{diagnostic_test}" for every rule.'
    )

    response = _CLIENT.chat.completions.create(
        model=_deployment,
        messages=[
            {"role": "system", "content": _RULE_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    text = response.choices[0].message.content or ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError(
                f"Could not parse LLM response as JSON for test '{diagnostic_test}'. "
                f"First 300 chars: {text[:300]}"
            )

    if isinstance(parsed, dict):
        raw_rules = next((v for v in parsed.values() if isinstance(v, list)), None)
        if raw_rules is None:
            raise ValueError(f"LLM returned a JSON object with no array for test '{diagnostic_test}'.")
    elif isinstance(parsed, list):
        raw_rules = parsed
    else:
        raise ValueError(f"LLM returned unexpected JSON type for test '{diagnostic_test}'.")

    valid_rules = []
    for rule in raw_rules:
        if not isinstance(rule, dict):
            continue
        if not {"raw_symptom", "node_type", "condition", "test", "source"}.issubset(rule):
            continue
        if rule["node_type"] not in VALID_NODE_TYPES:
            rule["node_type"] = "Symptom"
        rule["test"] = diagnostic_test  # enforce exact test name
        valid_rules.append(rule)

    return valid_rules


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Append rules to guideline_rules.json (skip already-present tests)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_guideline_rules(
    tests: list[str],
    rules_json_path: str,
) -> dict[str, int]:
    """For each test not already covered in guideline_rules.json, generate rules
    and append them.  Returns a mapping of test → number of rules added (0 = skipped)."""

    with open(rules_json_path, "r") as f:
        existing = json.load(f)

    existing_tests: set[str] = {
        r["test"] for r in existing if isinstance(r, dict) and "test" in r
    }

    added_counts: dict[str, int] = {}

    for test in tests:
        if test in existing_tests:
            print(f"  [skip] '{test}' already in guideline_rules.json")
            added_counts[test] = 0
            continue

        print(f"  [llm]  Generating rules for '{test}' …", end=" ", flush=True)
        try:
            rules = _generate_rules_for_test(test)
        except Exception as exc:
            print(f"FAILED ({exc})")
            added_counts[test] = 0
            continue

        existing.append({"_comment": f"PATHWAY: {test} — LLM-generated via kg_enrichment_pipeline"})
        existing.extend(rules)
        existing_tests.add(test)
        added_counts[test] = len(rules)
        print(f"{len(rules)} rules added.")

    with open(rules_json_path, "w") as f:
        json.dump(existing, f, indent=2)

    return added_counts


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Rebuild KG (calls build_kg.generate_knowledge_graph())
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_graph() -> dict:
    """Rebuild the PKL with full API enrichment.  Returns the stats dict."""
    print("Rebuilding knowledge graph (API enrichment in progress) …")
    stats = generate_knowledge_graph()
    print(f"  Graph rebuilt: {stats['nodes']} nodes, {stats['edges']} edges → {stats['pkl_path']}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Semantic mapping: CSV dx → KG Condition nodes
# ─────────────────────────────────────────────────────────────────────────────

_MAPPING_SYSTEM_PROMPT = (
    "You are a clinical NLP assistant. Your job is to map a raw patient diagnosis "
    "to the closest matching official condition from a provided list. "
    "If there is no medically reasonable match, return null. "
    "You must respond in strictly valid JSON format with a single key: 'matched_condition'."
)


def _semantic_match(raw_diagnosis: str, available_conditions: list[str]) -> str | None:
    """Return the best matching condition label from *available_conditions*, or None."""
    user_prompt = (
        f"Raw Patient Diagnosis: '{raw_diagnosis}'\n"
        f"Available Official Conditions: {json.dumps(available_conditions)}\n\n"
        "Return the closest matching Official Condition exactly as it appears in the list."
    )
    try:
        response = _CLIENT.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": _MAPPING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("matched_condition")
    except Exception as exc:
        print(f"  [!] LLM mapping failed for '{raw_diagnosis}': {exc}")
        return None


def _checkpoint_path(csv_input_path: str) -> str:
    """Return a sidecar checkpoint file path next to the input CSV."""
    base = os.path.splitext(os.path.abspath(csv_input_path))[0]
    return base + "_mapping_checkpoint.json"


def _load_checkpoint(path: str) -> dict[str, str | None]:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        print(f"  [checkpoint] Resuming from '{path}' ({len(data)} entries already mapped).")
        return data
    return {}


def _save_checkpoint(path: str, mapping: dict[str, str | None]) -> None:
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def map_csv_to_graph(
    csv_input_path: str,
    graph_pkl_path: str,
    csv_output_path: str,
) -> pd.DataFrame:
    """Enrich *csv_input_path* with matched conditions and linked tests, write to
    *csv_output_path*, and return the resulting DataFrame.

    A checkpoint file (<input_csv>_mapping_checkpoint.json) is written after
    every successful LLM call so the mapping step can be resumed if interrupted.
    The checkpoint is deleted on clean completion.
    """

    print(f"Loading knowledge graph from '{graph_pkl_path}' …")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)

    available_conditions = [
        node.replace("Condition: ", "")
        for node in kg.nodes
        if str(node).startswith("Condition: ")
    ]
    print(f"  {len(available_conditions)} condition nodes available for matching.")

    df = pd.read_csv(csv_input_path)
    if "dx" not in df.columns:
        raise ValueError("Input CSV must contain a 'dx' column.")

    unique_diagnoses = df["dx"].dropna().unique().tolist()
    print(f"  {len(unique_diagnoses)} unique diagnoses in CSV to map.")

    ckpt_path = _checkpoint_path(csv_input_path)
    mapping: dict[str, str | None] = _load_checkpoint(ckpt_path)

    remaining = [d for d in unique_diagnoses if d not in mapping]
    print(f"  {len(remaining)} diagnoses left to map (skipping {len(mapping)} already done).")

    print("Starting semantic matching (Azure OpenAI) …")
    for raw in remaining:
        matched = _semantic_match(raw, available_conditions)
        mapping[raw] = matched
        _save_checkpoint(ckpt_path, mapping)
        print(f"  '{raw}' → '{matched}'")

    df["matched_graph_condition"] = df["dx"].map(mapping)

    def _tests_for_condition(cond: str | None) -> str:
        if not cond:
            return "No match found"
        node_name = f"Condition: {cond}"
        if node_name not in kg.nodes:
            return "No linked tests found"
        tests = [
            tgt.replace("Test: ", "")
            for _, tgt, ed in kg.out_edges(node_name, data=True)
            if ed.get("relationship") == "REQUIRES_TEST"
        ]
        return ", ".join(tests) if tests else "No linked tests found"

    df["potential_tests"] = df["matched_graph_condition"].apply(_tests_for_condition)

    df.to_csv(csv_output_path, index=False)
    print(f"\nEnriched CSV saved to '{csv_output_path}'")

    # Clean up checkpoint — mapping completed successfully.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"Checkpoint '{ckpt_path}' removed.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KG enrichment + CSV diagnosis-mapping pipeline.")
    p.add_argument(
        "--input-csv", required=True,
        help="Path to the input CSV file containing a 'dx' column.",
    )
    p.add_argument(
        "--output-csv", default="patient_diagnoses_enriched.csv",
        help="Path for the enriched output CSV (default: patient_diagnoses_enriched.csv).",
    )
    p.add_argument(
        "--graph-json",
        default=os.path.join(_THIS_DIR, "diagnosis_to_tests_graph_data.json"),
        help="Path to diagnosis_to_tests_graph_data.json.",
    )
    p.add_argument(
        "--rules-json",
        default=os.path.join(_KG_DIR, "guideline_rules.json"),
        help="Path to guideline_rules.json.",
    )
    p.add_argument(
        "--graph-pkl",
        default=os.path.join(_KG_DIR, "triage_knowledge_graph.pkl"),
        help="Path to triage_knowledge_graph.pkl.",
    )
    p.add_argument(
        "--skip-rebuild", action="store_true",
        help="Skip LLM rule generation + graph rebuild (steps 1–4). Use existing PKL.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.skip_rebuild:
        # ── Step 1: extract tests ────────────────────────────────────────────
        print(f"\n[Step 1] Extracting diagnostic tests from '{args.graph_json}' …")
        tests = extract_unique_tests(args.graph_json)
        print(f"  {len(tests)} unique tests found: {tests}")

        # ── Step 2-3: generate + append rules ────────────────────────────────
        print(f"\n[Step 2-3] Generating and appending rules to '{args.rules_json}' …")
        counts = enrich_guideline_rules(tests, args.rules_json)
        total_new = sum(counts.values())
        print(f"  Total new rules appended: {total_new}")

        # ── Step 4: rebuild graph ─────────────────────────────────────────────
        print("\n[Step 4] Rebuilding knowledge graph …")
        rebuild_graph()
    else:
        print("[Steps 1-4 skipped] Using existing PKL.")

    # ── Step 5: CSV mapping ───────────────────────────────────────────────────
    print(f"\n[Step 5] Mapping CSV diagnoses → graph conditions …")
    result_df = map_csv_to_graph(args.input_csv, args.graph_pkl, args.output_csv)

    print("\n--- Preview (dx | matched_graph_condition | potential_tests) ---")
    print(result_df[["dx", "matched_graph_condition", "potential_tests"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
