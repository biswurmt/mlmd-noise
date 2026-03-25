# MLMD Presentation — Desk Reference

## Main Presentation Timeline

| Time | Slide | Section | Speaker | Notes |
|------|-------|---------|---------|-------|
| 0:00 | 1 | Title | Tyler | Names, advisor, course |
| 0:30 | 2–3 | Clinical Motivation + Data | Tyler | MLMD system, label noise problem, 2-track strategy |
| 2:15 | 4 | KG System Overview | Alice | Pipeline diagram |
| 3:15 | 5 | KG Construction | Alice | 45 rules, 10-step enrichment |
| 5:15 | 6 | Audit Pipeline | Alice | 4-pass: filter → normalize → ground → verify |
| 6:45 | 7 | Patient Mapping | Alice | Phase 1: extract; Phase 2: enrich + map |
| 7:45 | 8 | KG Results | Alice | Coverage, qualitative examples, audit outcomes |
| 9:15 | 9 | LiLAW Track | Tyler | alpha/beta/delta parameters; bilevel optimization |
| 11:15 | 10 | Evaluation Plan | Tyler | 4 conditions, metrics (recall, precision, ECE) |
| 12:00 | 11 | Next Steps & Limitations | Tyler | Roadmap, risks (hallucination, data shift, LiLAW unproven) |
| 13:00 | 12 | Summary | Either | Problem → solution → progress → path forward |

**Pacing target:** 13–15 min main deck + 5 min Q&A

---

## Appendix Slides Reference

| Slide | Topic | Use Case |
|-------|-------|----------|
| 14 | Limitations | LLM hallucination, data shift, clinical validation pending |
| 15 | Audit Pipeline Detail | Evidence grounding (PMC tiers), LLM verification criteria |
| 16 | Knowledge Graph Schema | Node types, edge types, metadata on every edge |
| 17 | Diagnotix Detail | Web app stack, node tooltip codes (HP/MONDO, ICD-10, LOINC) |
| 18 | Full Enrichment Table | All 11 enrichment steps (EMBL-EBI, UMLS, Infoway, Europe PMC, ClinicalTrials.gov) |
| 19 | Label Imputation Pipeline | Phase 1 & 2 detail, standalone patient mapper |

---

## Q&A Troubleshooting

**Q: "Won't LLMs hallucinate bad rules?"**
→ **Slide 14 (Limitations) + Slide 15 (Audit Detail)**
_Say:_ "That's the main risk we address. Our audit pipeline has 4 passes: we pre-filter, normalize to standard codes, ground in literature (PMC), then have an LLM reviewer check each rule against evidence tiers. Rules with no evidence get removed by default."

**Q: "How does the knowledge graph actually work?"**
→ **Slide 16 (KG Schema)**
_Say:_ "Three main node types—symptom, condition, test—connected by edges. Every edge has metadata: source (guideline), URL, and evidence weight from literature. So when a patient comes in with a diagnosis, we map it to the graph and recommend the downstream tests."

**Q: "Walk me through the enrichment—how many steps?"**
→ **Slide 18 (Full Enrichment Table)**
_Say:_ "Eleven steps. We start with curated rules, then add standard codes (HP, MONDO, SNOMED CT, LOINC, ICD-10). Then steps 8–10 add real-world evidence: literature co-occurrence counts from Europe PMC and trial counts from ClinicalTrials.gov. Those counts drive visualization width."

**Q: "How do you recover missing test labels from the EHR?"**
→ **Slide 19 (Label Imputation Pipeline)**
_Say:_ "Phase 1: we extract unique diagnoses from the `dx` column and ask the LLM what tests they warrant. Phase 2: we add those test pathways to the graph, re-enrich everything, then map every patient's diagnosis to recommended tests. If they should have had a test but didn't, we flag it as a recovered label."

**Q: "Isn't LiLAW unproven on clinical data?"**
→ **Slide 14 (Limitations)**
_Say:_ "Yes—that's our biggest remaining risk. LiLAW has been validated on standard label-noise benchmarks but not on real clinical asymmetric noise. That's why we're comparing four conditions: baseline, imputed-only, LiLAW-only, and combined. The evaluation will tell us if it helps."

**Q: "Can you show me the Diagnotix web app?"**
→ **Slide 17 (Diagnotix Detail)**
_Say:_ "It's a full-stack app (FastAPI + React). You enter any test name, the LLM generates 8–15 evidence-based triage rules, we run them through the audit pipeline, then add them to the graph. Every tooltip on a node shows the medical codes: HP/MONDO, ICD-10, LOINC, etc."
