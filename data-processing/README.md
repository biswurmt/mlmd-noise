# data-processing

This folder contains three scripts that form a two-phase pipeline for ingesting
real patient data into the Diagnotix knowledge graph.

**Phase 1** converts a raw patient diagnosis CSV into a structured list of
diagnostic tests (`csv_extract_potential_test.py`).
**Phase 2** uses that list to expand the knowledge graph with new triage rules
and then maps every patient diagnosis back to a graph condition with its
recommended tests (`kg_enrichment_pipeline.py`).
`csv_mapping.py` is a standalone utility that covers Phase 2's mapping step
alone, without rebuilding the graph.

---

## Files

### `csv_extract_potential_test.py`

**Purpose:** Given a patient CSV with a `dx` (diagnosis) column, queries Azure
OpenAI in batches to identify the standard diagnostic tests for each unique
diagnosis, and writes the results to a JSON file.

**What it does:**
1. Reads the CSV and collects all unique diagnosis strings from the `dx` column.
2. Sends them to Azure OpenAI in configurable batches (default 20 per call).
3. The LLM consolidates and standardises test names (e.g. all variations of
   "US Appendix" become `"Appendix Ultrasound"`) so they are suitable as graph
   node labels.
4. Saves a JSON file — `diagnoses_to_tests_graph_data.json` — with the
   structure:
   ```json
   [
     {
       "diagnosis": "Acute Appendicitis",
       "diagnostic_tests": ["Appendix Ultrasound", "Complete Blood Count (CBC)"]
     }
   ]
   ```

**Output:** `diagnoses_to_tests_graph_data.json` — consumed by
`kg_enrichment_pipeline.py` in Phase 2.

**Usage:**
```bash
# Edit INPUT_CSV and OUTPUT_JSON at the bottom of the file, then:
python csv_extract_potential_test.py
```

**Environment variables required** (in `data-processing/.env`):
```
ENDPOINT_URL=<Azure OpenAI endpoint>
DEPLOYMENT_NAME=<model deployment name>
```

---

### `csv_mapping.py`

**Purpose:** Standalone script that semantically maps raw patient diagnoses from
a CSV to condition nodes already present in the knowledge graph, then looks up
the tests linked to each matched condition.

**What it does:**
1. Loads the serialised knowledge graph PKL and extracts all `Condition:` nodes.
2. Reads unique diagnoses from the `dx` column of the input CSV.
3. For each diagnosis, calls Azure OpenAI to find the closest matching condition
   in the graph (fuzzy/semantic match — handles abbreviations, synonyms, and
   minor wording differences).
4. Traverses the graph's `REQUIRES_TEST` edges to retrieve the recommended
   diagnostic tests for each matched condition.
5. Writes an enriched CSV with two new columns:
   - `matched_graph_condition` — the best-matching condition node label
   - `potential_tests` — comma-separated list of tests linked via `REQUIRES_TEST`

**Use this script when** the knowledge graph is already up to date and you only
need to map a new patient CSV against it (i.e. you do not need to add new
pathways).

**Usage:**
```bash
# Edit INPUT_CSV, OUTPUT_CSV, and GRAPH_PKL at the bottom of the file, then:
python csv_mapping.py
```

**Environment variables required** (in `data-processing/.env`):
```
ENDPOINT_URL=<Azure OpenAI endpoint>
DEPLOYMENT_NAME=<model deployment name>
```

---

### `kg_enrichment_pipeline.py`

**Purpose:** End-to-end pipeline that combines graph expansion with patient
data mapping. Takes a patient CSV and the JSON produced by
`csv_extract_potential_test.py`, adds any missing diagnostic test pathways to the
knowledge graph, rebuilds the graph with full ontology enrichment, and then
maps every patient diagnosis to a condition and its recommended tests.

**What it does (5 steps):**

| Step | Action |
|------|--------|
| 1 | Read `diagnosis_to_tests_graph_data.json` and extract all unique test names. |
| 1b *(optional)* | If `--top-x X` is provided, send the candidate tests to the LLM in a single call asking it to rank them by clinical importance and return only the top X. Tests already in the graph are excluded before ranking. |
| 2 | For each selected test not already in `guideline_rules.json`, call Azure OpenAI to generate 8–15 triage rules grounded in authoritative clinical guidelines. |
| 3 | Append the new rules to `guideline_rules.json` (tests already present are skipped). |
| 4 | Rebuild the knowledge graph PKL via `build_kg.generate_knowledge_graph()` — this runs the full multi-ontology enrichment (SNOMED CT, ICD-10, LOINC, Europe PMC weights, etc.). |
| 5 | Semantically map every unique `dx` value in the input CSV to a graph condition node (with checkpoint/resume support), then write an enriched output CSV. |

**Checkpoint / resume:** The mapping step (Step 5) writes a sidecar file
`<input_csv>_mapping_checkpoint.json` after every successful LLM call. If the
script is interrupted, re-running it will skip already-mapped diagnoses and
continue from where it left off. The checkpoint is deleted on clean completion.

**Usage:**
```bash
python kg_enrichment_pipeline.py \
    --input-csv  /path/to/student_data.csv \
    --output-csv patient_diagnoses_enriched.csv

# Skip graph rebuild if the PKL is already current:
python kg_enrichment_pipeline.py \
    --input-csv  /path/to/student_data.csv \
    --output-csv patient_diagnoses_enriched.csv \
    --skip-rebuild
```

**All CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-csv` | *(required)* | Input CSV with a `dx` column |
| `--output-csv` | `patient_diagnoses_enriched.csv` | Path for the enriched output CSV |
| `--graph-json` | `data-processing/diagnosis_to_tests_graph_data.json` | Output of `csv_extract_potential_test.py` |
| `--rules-json` | `knowledge-graphs/guideline_rules.json` | Triage rules file to extend |
| `--graph-pkl` | `knowledge-graphs/triage_knowledge_graph.pkl` | Serialised graph to read/write |
| `--skip-rebuild` | `false` | Skip Steps 1–4; use the existing PKL |
| `--top-x X` | *(all tests)* | Ask the LLM to rank new candidate tests and add only the top X most clinically important ones to the graph |

**Environment variables required** (in `data-processing/.env` or
`knowledge-graphs/.env`):
```
ENDPOINT_URL=<Azure OpenAI endpoint>
DEPLOYMENT_NAME=<model deployment name>
```

---

## Typical end-to-end workflow

```bash
# Step A — extract tests from the patient CSV (produces the JSON)
python csv_extract_potential_test.py

# Step B — expand the graph and map patients (full pipeline)
python kg_enrichment_pipeline.py \
    --input-csv /path/to/student_data.csv \
    --output-csv patient_diagnoses_enriched.csv
```

If the graph is already up to date (e.g. you only received a new CSV batch):

```bash
python kg_enrichment_pipeline.py \
    --input-csv /path/to/new_batch.csv \
    --output-csv new_batch_enriched.csv \
    --skip-rebuild
```

---

## Input CSV format

All three scripts expect the input CSV to contain a column named **`dx`** with
raw patient diagnosis strings. All other columns are passed through unchanged to
the output.

---

## Output columns added to the CSV

| Column | Description |
|--------|-------------|
| `matched_graph_condition` | Label of the best-matching `Condition:` node in the knowledge graph, or blank if no reasonable match was found |
| `potential_tests` | Comma-separated diagnostic tests linked to that condition via `REQUIRES_TEST` edges in the graph |
