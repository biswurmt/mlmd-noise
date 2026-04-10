# backend/services

Business logic layer. Each module is imported by one or more routers.

## Files

### `chat_service.py`

Clinical decision-support chat backed by three evidence sources:

1. **KG context** — the current pathway's nodes and edges are serialised as a structured text block and injected into the system prompt. Each node includes its type, ontology codes (LOINC, ICD-10, SNOMED, EBI/HP), synonyms, and co-occurrence article counts.
2. **Qdrant RAG** — the user's message is first rewritten into 1–3 focused clinical search queries by a fast Kimi-K2.5-fast call, then each query hits the Qdrant vector DB (`query_guidelines` from `vector_db/build_vector_db.py`). Results are deduplicated and injected as numbered guideline passages.
3. **Semantic Scholar** — the same rewritten queries are sent to the Semantic Scholar API. Abstracts are numbered starting after the last RAG passage so all inline citations `[N]` stay unique.

The final LLM call uses **Nebius AI Studio — `moonshotai/Kimi-K2.5-fast`** regardless of `LLM_PROVIDER`.

`sync_chat(req)` is the synchronous entry point; routers call it via `asyncio.to_thread`.

**Required env:** `NEBIUS_API_KEY`

---

### `graph_service.py`

Loads `triage_knowledge_graph_enriched.pkl` (or whichever PKL is set in `KG_PKL_FILE`) and
converts the NetworkX `DiGraph` to JSON-safe Python dicts for FastAPI to return.

Key responsibilities:
- Normalises non-JSON-safe values: `NaN` → `None`, numpy integers/floats → Python primitives, numpy arrays → lists.
- Renames the edge `"source"` attribute (guideline name) to `"guideline_source"` so it does not collide with the react-force-graph `"source"` field (source node ID).
- Supports server-side pathway filtering: builds the 1-hop subgraph around the requested `Diagnostic_Test` node before serialising.
- Exposes a `"node_type"` alias for the `"type"` attribute so the frontend can use a consistent key across PKL versions.

Exported functions used by other modules:
- `load_graph_json(pathway)` — full graph or 1-hop subgraph as `{nodes, edges}`
- `load_test_nodes()` — `[{id, label}]` for every `Diagnostic_Test` node
- `get_existing_node_ids()` — set of all node IDs (used by `kg_service` to compute the diff)

**Required env:** `KG_PKL_FILE` (optional, defaults to `triage_knowledge_graph_enriched.pkl`)

---

### `kg_service.py`

Orchestrates the add-test pipeline triggered by `POST /api/add_test`:

1. **Snapshot** — records existing node IDs via `get_existing_node_ids()`.
2. **RAG grounding** — retrieves up to 5 guideline passages from Qdrant for the test name.
3. **Semantic Scholar grounding** — fetches 3 paper abstracts for additional context.
4. **LLM generation** — calls the configured provider (Nebius or Azure OpenAI) with a structured system prompt to generate 8–15 triage rules as a JSON array.
5. **Schema validation** — checks required fields, `node_type` whitelist, and `source` whitelist (≥30 recognised guideline bodies). Drops malformed or unrecognised rules.
6. **Convergence loop** (up to 3 iterations):
   - Normalises condition/symptom names via ICD-10-CM and HP/MONDO ontologies (`_normalise_rule` from `audit_guidelines`).
   - Verifies surviving rules with a second LLM call acting as a clinical reviewer (3 checks: source accuracy, clinical plausibility, condition specificity).
   - Regenerates targeted replacements for dropped (condition, node_type) pairs and re-enters the loop until stable or max iterations reached.
7. **Append** — writes verified rules to `guideline_rules.json` under a `_comment` section header.
8. **Rebuild** — calls `generate_knowledge_graph()` from `build_kg.py` to rebuild `triage_knowledge_graph.pkl`, then `_merge_base_into_enriched()` to sync new nodes into the enriched PKL without losing existing enrichment metadata.
9. **Diff** — computes new nodes and edges against the pre-rebuild snapshot and returns an `AddTestResponse`.

**Required env:** `LLM_PROVIDER` (`nebius` or `azure`), `NEBIUS_API_KEY` or `ENDPOINT_URL` + `DEPLOYMENT_NAME`.
**Optional env:** `BING_SEARCH_KEY` (enables live Bing Search grounding in rule generation).

---

### `semantic_scholar.py`

Thin client for the Semantic Scholar Academic Graph API (`/graph/v1/paper/search`).

- `search_papers(query, limit)` — keyword search, returns a tuple of paper dicts (hashable for `lru_cache`). Papers without abstracts are skipped. Rate-limited to ~1 req/s (free tier).
- `get_paper_by_id(paper_id)` — fetch a single paper by any Semantic Scholar ID format.
- `format_abstracts_section(papers, heading, start_index)` — formats results as a numbered block for LLM prompt injection. `start_index` lets callers continue numbering after RAG passage citations.

Each paper dict contains: `title`, `abstract` (truncated to 400 chars), `year`, `citation_count`, `doi`, `s2_id`, `authors`, `url`.

**Optional env:** `SEMANTIC_SCHOLAR_API_KEY` (raises rate limit from ~100 req/5 min to a higher tier).
