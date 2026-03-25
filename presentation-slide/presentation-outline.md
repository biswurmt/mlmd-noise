# Presentation Outline (Working Draft)
**LMP2392H | Alice Chua + Tyler Biswurm | ~15 min + 5 min Q&A**

---

## 1. Title (30s) — *Tyler*
- Title, names, advisor, course

## 2. Clinical Motivation (2 min) — *Tyler*
- MLMD at SickKids: parallel diagnostics → reduce ED wait times
- Label noise problem: negatives ambiguous (external tests invisible to EHR)
- Consequence: model learned chest pain doesn't warrant ECG → recall artificially depressed
- Our strategy: **two tracks** — clean the labels (Alice) + robust training (Tyler)

## 3. Data & Setting (45s) — *Tyler*
- SickKids ED EHR; triage-time features; asymmetric noise
- ICD-10 codes available post-assessment (relabeling only, not model inputs)

---

## 4. KG Track: System Overview (1 min) — *Alice*
- Pipeline diagram (from `draft-kg-pres.tex` §System Architecture)
- Rule Store → Graph Builder → Graph Store → Patient Mapper / Diagnotix
- **Compress or cut**: Diagnotix (mention only, or last slide)

## 5. KG Track: Knowledge Graph Construction (2 min) — *Alice*
- 45 curated clinical rules: `raw_symptom → condition → test` (JSON example)
- 10-step ontology enrichment (SNOMED CT CA, LOINC, ICD-10, ICD-10-CA, HP/MONDO, RxNorm, Europe PMC, ClinicalTrials.gov)
- Graph schema: node types + edge types (can show as a summary table, compress the full schema slide)

## 6. KG Track: Audit Pipeline (1.5 min) — *Alice*
- Why: LLM rules can be hallucinated/misattributed
- 4-pass pipeline: pre-process → normalise → evidence grounding (PMC tiers) → LLM verification
- Output: clean rule store fed back to graph builder

## 7. KG Track: Patient Mapping Pipeline (1 min) — *Alice*
- Phase 1: GPT-4o Nano extracts diagnosis → test mapping from `dx` column
- Phase 2: enrich graph + map patients to recommended tests
- Show example output / coverage stats if available

## 8. KG Track: Results & Validation (1.5 min) — *Alice*
- Coverage: X diagnoses mapped, Y tests recovered
- Example recovered labels (qualitative)
- Audit pipeline outcomes: rules retained/removed by tier
- What's pending: quantitative label recovery benchmark, clinician review

---

## 9. LiLAW Track (2 min) — *Tyler*
- What LiLAW does: learns (α, β, δ) to down-weight suspected false negatives via bilevel optimization
- Complements KG imputation: handles probabilistic cases KG can't adjudicate
- **Implementation status**: what's done, what's next, why behind
- Integration plan: KG-relabeled data → LiLAW training → retrained MLMD

---

## 10. Evaluation Plan (45s) — *Tyler*
- Baselines: production MLMD / imputed-only / LiLAW-only / combined
- Metrics: recall (efficiency), precision (safety), PR curves, ECE
- Three-phase validation: map audit → label recovery benchmark → blinded clinician review

## 11. Next Steps & Limitations (1 min) — *Tyler*
- Complete LiLAW implementation; retrain all four conditions
- LLM hallucination risk (audit pipeline mitigates); data shift; LiLAW unproven on clinical data
- Final report due ~3 weeks

## 12. Summary (30s) — *Either*
- Problem → two-track solution → KG complete, LiLAW in progress → clear path to evaluation

---

## Speaker Time Summary

| Speaker | Sections | Approx. Time |
|---------|----------|--------------|
| Tyler | 1–3, 9–12 | ~6 min |
| Alice | 4–8 | ~7 min |

---

## Compression Notes
- **Cut or 1-slide**: Diagnotix web app (mention as tooling, not a full slide unless time permits)
- **Compress**: Graph schema (fold into KG Construction slide as a table)
- **Compress**: 10-step enrichment table (highlight 3-4 key ontologies, don't read every row)
- **Expand if time**: KG results / example recovered labels — this is the most tangible output
