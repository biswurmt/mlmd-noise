# Diagnotix: Clinical Intelligence at the Point of Decision

**Team:** The Wanderers  
**Event:** BioxAI Hackathon 2026  
**Project:** Diagnotix  
**Members:** Alice Chua (ML + Clinical Informatics), Tyler Biswurm (Systems + Full-Stack)  
**Institution:** University of Toronto  
**Date:** March 29, 2026  
**Repository:** https://github.com/biswurmt/mlmd-noise

---

The world will be short 4.3 million doctors by 2030. Medical school takes a decade, costs a fortune, and produces graduates who cannot be everywhere at once. Meanwhile populations age, hospitals fill, and clinical expertise remains trapped in the heads of people who must sleep, retire, and occasionally quit.

Generative AI promised a fix. It delivered instead a cautionary tale: systems that cite papers no one wrote, recommend drugs no one should take, and provide no way to check their work. Clinicians noticed. They stopped using them.

Diagnotix takes a different approach. We start with what doctors actually trust—primary clinical guidelines from ACR, AHA/ACC, and NICE—and encode them into a living knowledge graph. We enrich this foundation with standard medical ontologies: SNOMED CT for clinical terms, ICD-10 for diagnoses, LOINC for lab values, RxNorm for medications. When our LLM pipeline proposes new clinical nodes, a multi-pass audit validates each against peer-reviewed literature before it enters production. The result is reasoning clinicians can verify, query, and stake decisions on.

Our first market is emergency department triage, where the arithmetic is brutal. EDs run at 100-120% capacity not because they lack beds but because they lack decisions fast enough. Triage nurses already collect vital signs, take histories, and order basic tests. Diagnotix amplifies that authority with specialist-level guidance on which tests to order when. Parallel processing replaces serial bottlenecks. Patient throughput increases. Physician workload decreases.

We know this works because we have already tried it. A parallel implementation at SickKids Hospital in Toronto earned positive feedback from ED physicians—a validation most healthtech startups spend years chasing.

From triage we expand to hospital systems, urgent care networks, and primary care. The endgame is licensing our knowledge graph as a clinical API to health-AI companies that need trustworthy medical grounding but cannot build it themselves. Clinical decision support markets grow 11% annually. The market for trustworthy clinical AI does not yet exist because no one has shipped a trustworthy product.

We will.
