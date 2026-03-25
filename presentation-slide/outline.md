# Presentation Outline (Working Draft)
**LMP2392H | Alice Chua + Tyler Biswurm | ~15 min + 5 min Q&A**

---

## 1. Title (30s) — *Tyler*
- Title, names, advisor, course

## 2. Clinical Motivation (2 min) — *Tyler*
- MLMD at SickKids: parallel diagnostics → reduce ED wait times
- Label noise problem: negatives ambiguous (external tests invisible to EHR)
- Consequence: model learned chest pain doesn't warrant ECG → recall artificially depressed
- Silent trial complete; clinical trial next

## 3. Data & Setting (45s) — *Tyler*
- 520k ED visits; triage-time features; asymmetric noise
- `dx` field: free-text post-assessment diagnoses — used by label imputation pipeline
- High transfer rate from community hospitals → primary source of false negatives

## 4. Relevant Work (1 min) — *Tyler*
- LLMs + KGs: Yao, Li, Cao, Han (Denoise2Impute)
- Noise-robust training: Yang, Ren, Northcutt, Moturu (LiLAW)

## 5. Research Questions (45s) — *Tyler*
- RQ1: Can a KG-grounded LLM pipeline recover false-negative labels, and does retraining improve recall?
- RQ2: Does noise-robust meta-learning improve recall under asymmetric EHR noise without explicit annotations?
- RQ3: Does combining both approaches yield gains beyond either alone?

## 6. Approach (30s) — *Tyler*
- Two complementary strategies: label imputation (deterministic cases) + robust training (probabilistic cases)
- *Handoff to Alice*

---

## 7. Label Imputation: Pipeline Overview (1 min) — *Alice*
- Rule Store → Graph Builder → Graph Store → Label Imputation / Diagnotix
- Audit pipeline feeds back to rule store

## 8. Label Imputation: Rules & Enriched Graph (1.5 min) — *Alice*
- 45 curated rules: `raw_symptom → condition → test` (JSON example)
- 10-step ontology enrichment (HP/MONDO, SNOMED CT CA, LOINC, ICD-10/CA, Europe PMC, ClinicalTrials.gov)

## 9. Label Imputation: Audit Pipeline (1.5 min) — *Alice*
- Why: LLM-generated rules can hallucinate or misattribute sources
- 4-pass: Filter & dedup → Normalise → Ground (Europe PMC tiers) → LLM Verify
- Output: clean rule store fed back to graph builder

## 10. Label Imputation: Diagnotix (30s) — *Alice*
- Web app: enter a test → system generates, audits, and adds rules to the KG
- *Handoff to Tyler*

---

## 11. Robust Training (2 min) — *Tyler*
- LiLAW: learns (α, β, δ) to down-weight suspected false negatives via bilevel optimisation
- Complements label imputation: handles probabilistic cases KG can't adjudicate
- Properties: only 3 extra params, no specially curated clean validation set required

## 12. Evaluation Plan (45s) — *Tyler*
- 4 conditions: production baseline / imputed-only / LiLAW-only / combined
- Metrics: recall (efficiency), precision (safety), PR curves, ECE
- Additional: label recovery benchmark; eventual clinician review

## 13. Progress & Roadmap (45s) — *Tyler*
- Done: rule store, enrichment pipeline, audit pipeline, Diagnotix
- Remaining: complete label imputation pipeline, LiLAW implementation, retrain, evaluate
- Final delivery: April 14

## 14. Thank You — *Either*
- Discussion prompts: limitations, audit pipeline, KG deep-dive, Diagnotix demo

---

## Speaker Time Summary

| Speaker | Sections | Approx. Time |
|---------|----------|--------------|
| Tyler | 1–6, 11–14 | ~8 min |
| Alice | 7–10 | ~5 min |

---

## Compression Notes
- **Compress**: 10-step enrichment table (highlight 3–4 key ontologies, don't read every row)
- **Cut or 1-slide**: Diagnotix (mention as tooling; demo in Q&A if time)
- **Expand if time**: label recovery benchmark results / example recovered labels — most tangible output
