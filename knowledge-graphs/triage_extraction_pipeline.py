"""
triage_extraction_pipeline.py
==============================
GraphRAG pipeline for Emergency Department triage decision support.

Pipeline stages
---------------
1. Synthea Mocker     — generate 4 synthetic patient triage records
2. Entity Extractor   — map patient data to knowledge-graph node categories
3. Graph Retrieval    — traverse the KG to find candidate tests + reasoning
4. LLM Prompt Assembly— format everything into a structured system prompt

Usage
-----
    python triage_extraction_pipeline.py                  # run all 4 patients, full output
    python triage_extraction_pipeline.py --output prompt  # assembled LLM prompts only
    python triage_extraction_pipeline.py --kg-path /path/to/triage_knowledge_graph.pkl
"""

import argparse
import json
import os
import pickle
import re

# =====================================================================
# SECTION 1: Synthea Mocker — 4 Synthetic Patient Triage Records
# =====================================================================

def generate_mock_patients() -> list:
    """
    Return a list of 4 synthetic triage patient records, one per target
    diagnostic pathway:
        P-001  ECG                (cardiac)
        P-002  Testicular Ultrasound
        P-003  Arm X-Ray          (FOOSH trauma)
        P-004  Appendix Ultrasound
    """
    return [
        {
            # ── Pathway 1: ECG ──────────────────────────────────────
            "patient_id": "P-001",
            "pathway":    "ECG",
            "demographics": {"age": 58, "sex": "M"},
            "vitals": {
                "heart_rate":       112,
                "bp_systolic":      165,
                "bp_diastolic":      95,
                "respiratory_rate":  20,
                "temperature":       37.1,
                "o2_saturation":     96,
            },
            "chief_complaint":    "Chest pain radiating to left arm with diaphoresis",
            "pain_scale":          8,
            "pain_onset":         "Sudden onset 45 minutes ago",
            "mechanism_of_injury": "",
            "clinical_attributes": [],
            "medical_history":    ["Hypertension", "Diabetes mellitus type 2", "Smoker"],
        },
        {
            # ── Pathway 2: Testicular Ultrasound ────────────────────
            "patient_id": "P-002",
            "pathway":    "Testicular Ultrasound",
            "demographics": {"age": 22, "sex": "M"},
            "vitals": {
                "heart_rate":       98,
                "bp_systolic":     128,
                "bp_diastolic":     78,
                "respiratory_rate": 16,
                "temperature":      37.4,
                "o2_saturation":    99,
            },
            "chief_complaint":    "Sudden onset severe left scrotal pain with nausea and vomiting",
            "pain_scale":          9,
            "pain_onset":         "Sudden onset approximately 2 hours ago",
            "mechanism_of_injury": "",
            "clinical_attributes": [],
            "medical_history":    ["No significant past medical history"],
        },
        {
            # ── Pathway 3: Arm X-Ray (FOOSH) ────────────────────────
            "patient_id": "P-003",
            "pathway":    "Arm X-Ray",
            "demographics": {"age": 34, "sex": "F"},
            "vitals": {
                "heart_rate":       88,
                "bp_systolic":     122,
                "bp_diastolic":     76,
                "respiratory_rate": 14,
                "temperature":      36.9,
                "o2_saturation":    99,
            },
            "chief_complaint":    "Right wrist pain and visible deformity after a fall",
            "pain_scale":          7,
            "pain_onset":         "Acute onset after fall approximately 1 hour ago",
            "mechanism_of_injury": "Fall on outstretched hand (FOOSH) from standing height",
            "clinical_attributes": [],
            "medical_history":    ["No significant past medical history"],
        },
        {
            # ── Pathway 4: Appendix Ultrasound ──────────────────────
            "patient_id": "P-004",
            "pathway":    "Appendix Ultrasound",
            "demographics": {"age": 24, "sex": "F"},
            "vitals": {
                "heart_rate":      102,
                "bp_systolic":     118,
                "bp_diastolic":     72,
                "respiratory_rate": 18,
                "temperature":      38.2,
                "o2_saturation":    98,
            },
            "chief_complaint":    "Periumbilical pain migrating to right lower quadrant with nausea",
            "pain_scale":          6,
            "pain_onset":         "Gradual onset over the past 12 hours",
            "mechanism_of_injury": "",
            "clinical_attributes": [
                "Rebound tenderness on RLQ palpation",
                "Pain migrating from periumbilical to RLQ",
            ],
            "medical_history": [
                "Last menstrual period 3 weeks ago",
                "No prior abdominal surgeries",
            ],
        },
    ]


# =====================================================================
# SECTION 2: Entity Extractor
# =====================================================================

# Keyword-to-graph-node-label mappings for each entity category.
# Keys are substrings searched in lowercased free-text fields.
# Values are the canonical raw_symptom labels used when building the KG.

_SYMPTOM_KEYWORDS: dict = {
    "chest pain":            "chest pain",
    "shortness of breath":   "shortness of breath",
    " sob ":                 "shortness of breath",
    "palpitation":           "palpitations",
    "syncope":               "syncope",
    "jaw pain":              "jaw pain",
    "left arm pain":         "left arm pain",
    "diaphoresis":           "diaphoresis",
    "sweating":              "diaphoresis",
    "scrotal pain":          "scrotal pain",
    "groin pain":            "groin pain",
    "testicular swelling":   "testicular swelling",
    "nausea":                "nausea",
    "vomit":                 "vomiting",
    "arm pain":              "arm pain",
    "wrist pain":            "wrist pain",
    "shoulder pain":         "shoulder pain",
    "deformity":             "visible limb deformity",
    "periumbilical":         "periumbilical pain",
    "right lower quadrant":  "right lower quadrant pain",
    " rlq ":                 "right lower quadrant pain",
    "abdominal pain":        "diffuse abdominal pain",
    "acute scrotal":         "acute scrotal pain",
    "atypical chest":        "atypical chest pain",
}

_RISK_FACTOR_KEYWORDS: dict = {
    "hypertension":    "hypertension",
    "diabetes":        "diabetes mellitus",
    "cardiac history": "prior cardiac history",
    "heart disease":   "prior cardiac history",
    "heart failure":   "prior cardiac history",
}

# Each key is a tuple of substrings that must ALL appear in the MOI text.
_MOI_KEYWORDS: dict = {
    ("fall", "outstretched"): "fall on outstretched hand",
    ("fall", "foosh"):        "fall on outstretched hand",
    ("crush",):               "crush injury arm",
    ("trauma", "scrotal"):    "recent scrotal trauma",
    ("trauma", "testicular"): "recent scrotal trauma",
    ("trauma", "groin"):      "recent scrotal trauma",
}

_CLINICAL_ATTR_KEYWORDS: dict = {
    "rebound":   "rebound tenderness",
    "migrating": "periumbilical pain migrating to rlq",
    "migrated":  "periumbilical pain migrating to rlq",
}


def extract_entities(patient_record: dict) -> dict:
    """
    Parse a patient triage record and extract entities that map directly
    to node categories in the NetworkX knowledge graph.

    Returns
    -------
    dict with keys:
        symptoms, vitals, demographics, risk_factors,
        mechanisms_of_injury, clinical_attributes
    Each value is a list of canonical raw_symptom label strings.
    """
    entities: dict = {
        "symptoms":             [],
        "vitals":               [],
        "demographics":         [],
        "risk_factors":         [],
        "mechanisms_of_injury": [],
        "clinical_attributes":  [],
    }

    # ── 1. Symptoms from chief complaint (substring search) ──────────
    complaint = f" {patient_record.get('chief_complaint', '').lower()} "
    for keyword, mapped in _SYMPTOM_KEYWORDS.items():
        if keyword in complaint and mapped not in entities["symptoms"]:
            entities["symptoms"].append(mapped)

    # ── 2. Vital sign threshold crossings ────────────────────────────
    vitals = patient_record.get("vitals", {})

    hr = vitals.get("heart_rate")
    if hr is not None:
        if hr > 120:
            entities["vitals"].append("heart rate > 120")
        elif hr > 100:
            # > 100 is the AHA/ACC threshold; only add > 120 if both apply
            entities["vitals"].append("heart rate > 100")
        if hr < 60:
            entities["vitals"].append("heart rate < 60")

    bp_sys = vitals.get("bp_systolic")
    if bp_sys is not None:
        if bp_sys > 180:
            entities["vitals"].append("systolic bp > 180")
        if bp_sys < 90:
            entities["vitals"].append("systolic bp < 90")

    temp = vitals.get("temperature")
    if temp is not None and temp > 37.3:
        entities["vitals"].append("fever > 37.3")

    # ── 3. Demographics ───────────────────────────────────────────────
    demo = patient_record.get("demographics", {})
    age  = demo.get("age")
    sex  = demo.get("sex", "").upper()

    if age is not None:
        if age > 35:
            entities["demographics"].append("age > 35")
        if age < 18:
            entities["demographics"].append("age < 18")
    if sex == "F" and age is not None and 12 <= age <= 55:
        entities["demographics"].append("female of childbearing age")

    # ── 4. Risk factors from medical history ──────────────────────────
    for condition_str in patient_record.get("medical_history", []):
        clow = condition_str.lower()
        for keyword, mapped in _RISK_FACTOR_KEYWORDS.items():
            if keyword in clow and mapped not in entities["risk_factors"]:
                entities["risk_factors"].append(mapped)

    # ── 5. Mechanism of injury ────────────────────────────────────────
    moi_text = patient_record.get("mechanism_of_injury", "").lower()
    if moi_text:
        for keywords, mapped_moi in _MOI_KEYWORDS.items():
            if all(kw in moi_text for kw in keywords):
                if mapped_moi not in entities["mechanisms_of_injury"]:
                    entities["mechanisms_of_injury"].append(mapped_moi)

    # ── 6. Clinical attributes (onset + explicit attribute list) ──────
    onset = patient_record.get("pain_onset", "").lower()
    if re.search(r"\bsudden\b|\bacute\b", onset):
        entities["clinical_attributes"].append("sudden pain onset")
    elif re.search(r"\bgradual\b|\bslow\b|\bprogressive\b", onset):
        entities["clinical_attributes"].append("gradual pain onset")

    for attr in patient_record.get("clinical_attributes", []):
        attr_lower = attr.lower()
        for keyword, mapped_attr in _CLINICAL_ATTR_KEYWORDS.items():
            if keyword in attr_lower and mapped_attr not in entities["clinical_attributes"]:
                entities["clinical_attributes"].append(mapped_attr)

    return entities


# =====================================================================
# SECTION 3: Graph Retrieval
# =====================================================================

def _upsert_test(
    recommended_tests: dict,
    test_name: str,
    test_node: str,
    guideline: str,
    trace_entry: dict,
) -> None:
    """Insert or update a test entry in the recommended_tests accumulator."""
    if test_name not in recommended_tests:
        recommended_tests[test_name] = {
            "test_node":          test_node,
            "guideline_sources":  set(),
            "triggering_entities":set(),
            "reasoning_trace":    [],
        }
    recommended_tests[test_name]["guideline_sources"].add(guideline)
    recommended_tests[test_name]["triggering_entities"].add(
        trace_entry["triggering_entity"]
    )
    recommended_tests[test_name]["reasoning_trace"].append(trace_entry)


def get_triage_context(
    extracted_entities: dict,
    kg_path: str = "triage_knowledge_graph.pkl",
) -> dict:
    """
    Load the triage knowledge graph and traverse it to identify recommended
    diagnostic tests for the supplied extracted entities.

    Traversal strategy
    ------------------
    For each graph node that fuzzy-matches an extracted entity:
        • DIRECTLY_INDICATES_TEST edges  → record the test directly
        • INDICATES_CONDITION edges      → follow the condition node's
          REQUIRES_TEST edges to collect the test one hop further

    The reasoning trace records the triggering entity, the traversal path,
    and the clinical guideline source attached to each edge.

    Parameters
    ----------
    extracted_entities : dict   Output from extract_entities()
    kg_path            : str    Path to the serialized NetworkX DiGraph (.pkl)

    Returns
    -------
    dict with:
        matched_graph_nodes  — sorted list of KG node IDs that were matched
        recommended_tests    — dict keyed by test name, each containing
                               guideline_sources, triggering_entities,
                               and reasoning_trace
    """
    if not os.path.exists(kg_path):
        raise FileNotFoundError(
            f"Knowledge graph not found at: '{kg_path}'. "
            "Run 'python build_kg.py' first."
        )

    with open(kg_path, "rb") as f:
        G = pickle.load(f)

    # Flatten all extracted entities into a single list
    all_entities = [
        item
        for category_items in extracted_entities.values()
        for item in category_items
    ]

    # ── Match entities → graph nodes (case-insensitive substring match) ──
    # The node ID format is "Prefix: Label", e.g. "Symptom: Chest Pain".
    # We strip the prefix and compare against the lowercased label.
    matched_nodes: set = set()
    for entity in all_entities:
        entity_lower = entity.strip().lower()
        for node_id in G.nodes():
            node_label = (
                node_id.split(": ", 1)[-1].lower()
                if ": " in node_id
                else node_id.lower()
            )
            if entity_lower in node_label or node_label in entity_lower:
                matched_nodes.add(node_id)

    # ── Graph traversal: entity → condition → test ───────────────────
    recommended_tests: dict = {}

    for node_id in matched_nodes:
        for _, neighbor, edge_data in G.out_edges(node_id, data=True):
            relationship = edge_data.get("relationship", "")
            guideline    = edge_data.get("source", "Unknown")

            if neighbor.startswith("Test:"):
                # Direct edge: entity → test
                test_name = neighbor.split(": ", 1)[-1]
                _upsert_test(
                    recommended_tests, test_name, neighbor, guideline,
                    {
                        "triggering_entity": node_id,
                        "relationship":      relationship,
                        "guideline":         guideline,
                    },
                )

            elif neighbor.startswith("Condition:"):
                # Two-hop: entity → condition → test
                condition_name = neighbor.split(": ", 1)[-1]
                for _, test_node, test_edge in G.out_edges(neighbor, data=True):
                    if test_node.startswith("Test:"):
                        test_name   = test_node.split(": ", 1)[-1]
                        t_guideline = test_edge.get("source", guideline)
                        _upsert_test(
                            recommended_tests, test_name, test_node, t_guideline,
                            {
                                "triggering_entity":     node_id,
                                "intermediate_condition":condition_name,
                                "relationship": (
                                    f"INDICATES_CONDITION({condition_name})"
                                    " → REQUIRES_TEST"
                                ),
                                "guideline": t_guideline,
                            },
                        )

    # Serialize sets for downstream JSON use
    for details in recommended_tests.values():
        details["guideline_sources"]   = sorted(details["guideline_sources"])
        details["triggering_entities"] = sorted(details["triggering_entities"])

    return {
        "matched_graph_nodes": sorted(matched_nodes),
        "recommended_tests":   recommended_tests,
    }


# =====================================================================
# SECTION 4: LLM Prompt Assembly
# =====================================================================

def assemble_llm_prompt(patient_record: dict, triage_context: dict) -> str:
    """
    Format the patient record and graph-retrieved diagnostic context into a
    clean, structured system prompt ready to be passed to a locally hosted
    LLM for the final clinical decision-support inference call.

    The prompt instructs the LLM to:
        1. Confirm or deprioritize each graph-recommended test
        2. Flag additional tests not covered by the KG
        3. Assign a CTAS triage level (1–5)
        4. Surface time-critical findings
        5. Return output in a strict JSON schema

    Parameters
    ----------
    patient_record  : dict   A single patient record from generate_mock_patients()
    triage_context  : dict   Output from get_triage_context()

    Returns
    -------
    str — fully formatted prompt string
    """
    demo   = patient_record.get("demographics", {})
    vitals = patient_record.get("vitals", {})
    history = patient_record.get("medical_history", ["None reported"])

    # ── Build the recommended-tests block ────────────────────────────
    if triage_context["recommended_tests"]:
        tests_block = ""
        for test_name, details in triage_context["recommended_tests"].items():
            sources_str = ", ".join(details["guideline_sources"])

            # Deduplicate trace lines
            trace_lines: set = set()
            for trace in details["reasoning_trace"]:
                entity    = trace.get("triggering_entity", "N/A")
                condition = trace.get("intermediate_condition")
                rel       = trace.get("relationship", "")
                guide     = trace.get("guideline", "")
                if condition:
                    trace_lines.add(
                        f"    • {entity} —[{rel}]— {test_name}  (Guideline: {guide})"
                    )
                else:
                    trace_lines.add(
                        f"    • {entity} —[{rel}]— {test_name}  (Guideline: {guide})"
                    )

            tests_block += (
                f"\n  ► {test_name}\n"
                f"    Guideline Sources : {sources_str}\n"
                f"    Reasoning Trace   :\n"
                + "\n".join(sorted(trace_lines))
                + "\n"
            )
    else:
        tests_block = "\n  No matching tests found in the knowledge graph.\n"

    # ── Matched nodes summary ─────────────────────────────────────────
    matched_nodes_str = (
        "\n".join(f"  • {n}" for n in triage_context["matched_graph_nodes"])
        or "  (none)"
    )

    # ── Vitals formatting ─────────────────────────────────────────────
    hr       = vitals.get("heart_rate",       "N/A")
    bp_s     = vitals.get("bp_systolic",      "N/A")
    bp_d     = vitals.get("bp_diastolic",     "N/A")
    rr       = vitals.get("respiratory_rate", "N/A")
    temp     = vitals.get("temperature",      "N/A")
    o2       = vitals.get("o2_saturation",    "N/A")

    moi_str  = patient_record.get("mechanism_of_injury") or "Not applicable"
    sep      = "=" * 70

    prompt = f"""\
SYSTEM PROMPT — Clinical Decision-Support AI (Emergency Triage Module)
{sep}

You are a clinical decision-support assistant embedded in an Emergency
Department triage workflow. Your outputs are ADVISORY ONLY and must be
reviewed and approved by a licensed clinician before any clinical action.

Ontology sources  : SNOMED CT US (UMLS), SNOMED CT CA (Infoway), ICD-10
Guidelines loaded : AHA/ACC · ACR · NICE · CTAS

{sep}
PATIENT PRESENTATION
{sep}
  Patient ID          : {patient_record.get('patient_id', 'N/A')}
  Age / Sex           : {demo.get('age', '?')} y/o {demo.get('sex', '?')}
  Chief Complaint     : {patient_record.get('chief_complaint', 'N/A')}
  Pain Scale          : {patient_record.get('pain_scale', 'N/A')} / 10
  Pain Onset          : {patient_record.get('pain_onset', 'N/A')}
  Mechanism of Injury : {moi_str}

VITAL SIGNS
  Heart Rate          : {hr} bpm
  Blood Pressure      : {bp_s}/{bp_d} mmHg
  Respiratory Rate    : {rr} breaths/min
  Temperature         : {temp} °C
  O2 Saturation       : {o2}%

MEDICAL HISTORY
{chr(10).join('  - ' + h for h in history)}

{sep}
KNOWLEDGE GRAPH RETRIEVAL RESULTS
{sep}
Matched graph nodes ({len(triage_context['matched_graph_nodes'])} total):
{matched_nodes_str}

Evidence-graded recommended diagnostic tests:
{tests_block}
{sep}
INSTRUCTIONS FOR LLM RESPONSE
{sep}
Using the patient presentation and knowledge-graph context above:

  1. CONFIRM or DEPRIORITIZE each recommended test with clinical justification.
  2. IDENTIFY any additional tests clinically indicated but absent from the
     graph retrieval (e.g., blood work, CT, MRI, urine hCG).
  3. ASSIGN a CTAS triage level (1 = Resuscitation … 5 = Non-urgent) with
     brief rationale.
  4. FLAG any time-critical findings requiring immediate intervention
     (e.g., suspected torsion within 6-hour window, STEMI activation).
  5. Provide a concise CLINICAL SUMMARY (2–4 sentences).

Respond ONLY in the following JSON schema — no prose outside the JSON:

{{
  "confirmed_tests": [
    {{
      "test":          "<test name>",
      "priority":      "<stat | urgent | routine>",
      "justification": "<clinical reasoning>"
    }}
  ],
  "additional_tests": [
    {{
      "test":   "<test name>",
      "reason": "<clinical reasoning>"
    }}
  ],
  "ctas_level":        <integer 1–5>,
  "ctas_rationale":    "<text>",
  "time_critical_flags": ["<flag 1>", "<flag 2>"],
  "clinical_summary":  "<text>"
}}
"""
    return prompt


# =====================================================================
# MAIN: Run the full pipeline for all 4 mock patients
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Triage GraphRAG Pipeline"
    )
    parser.add_argument(
        "--kg-path",
        default="triage_knowledge_graph_enriched.pkl",
        help="Path to the serialized KG .pkl file (default: triage_knowledge_graph_enriched.pkl)",
    )
    parser.add_argument(
        "--output",
        choices=["entities", "context", "prompt", "all"],
        default="all",
        help="Control what is printed: entities, context JSON, LLM prompt, or all (default: all)",
    )
    args = parser.parse_args()

    patients = generate_mock_patients()
    print(f"Generated {len(patients)} synthetic patient records.\n")

    for patient in patients:
        sep = "=" * 70
        print(sep)
        print(
            f"Patient {patient['patient_id']} "
            f"({patient['demographics']['age']}y {patient['demographics']['sex']}) "
            f"— Target pathway: {patient['pathway']}"
        )
        print(sep)

        # ── Stage 1: Entity extraction ────────────────────────────────
        entities = extract_entities(patient)

        if args.output in ("entities", "all"):
            print("\n[Stage 1] Extracted entities:")
            print(json.dumps(entities, indent=2))

        # ── Stage 2: Graph retrieval ──────────────────────────────────
        try:
            context = get_triage_context(entities, kg_path=args.kg_path)
        except FileNotFoundError as exc:
            print(f"\n  ERROR: {exc}\n")
            continue

        if args.output in ("context", "all"):
            print("\n[Stage 2] Graph retrieval context:")
            # Trim reasoning_trace to avoid console flood in 'all' mode
            display_context = {
                "matched_graph_nodes": context["matched_graph_nodes"],
                "recommended_tests": {
                    k: {
                        "guideline_sources":   v["guideline_sources"],
                        "triggering_entities": v["triggering_entities"],
                        "reasoning_trace":     v["reasoning_trace"][:3],  # first 3 only
                    }
                    for k, v in context["recommended_tests"].items()
                },
            }
            print(json.dumps(display_context, indent=2))

        # ── Stage 3: LLM prompt assembly ──────────────────────────────
        prompt = assemble_llm_prompt(patient, context)

        if args.output in ("prompt", "all"):
            print("\n[Stage 3] Assembled LLM prompt:")
            print(prompt)

        print()
