# Diagnotix — AI-Powered Emergency Triage Decision Support

Diagnotix is a GraphRAG-based clinical decision-support system for Emergency Department triage. It combines a multi-ontology knowledge graph — built from AHA/ACC, ACR, NICE, and CTAS guidelines — with a vector database of medical guidelines and a live Semantic Scholar feed to give ED clinicians grounded, evidence-cited reasoning at the point of care.

---

## Repository Structure

| Folder | What's in it |
|---|---|
| [`diagnotix/`](diagnotix/README.md) | Full-stack web application — FastAPI backend + React 19 frontend for exploring and expanding the triage knowledge graph |
| [`knowledge-graphs/`](knowledge-graphs/README.md) | KG construction pipeline, ontology enrichment, audit tools, vector DB indexer, and GraphRAG triage pipeline |
| [`data-processing/`](data-processing/README.md) | Scripts for ingesting real patient CSVs — extracts diagnostic tests and maps diagnoses back to KG conditions |
| [`comms/`](comms/README.md) | Course presentation (LaTeX/Beamer) and pitch deck materials |

---

## Diagnostic Pathways

The knowledge graph encodes triage rules for four diagnostic pathways drawn from four clinical guidelines:

| Pathway | Guidelines | Target conditions |
|---|---|---|
| ECG | AHA/ACC, CTAS | Acute MI, Arrhythmia, Tachycardia, Cardiogenic Shock |
| Testicular Ultrasound | ACR, NICE | Testicular Torsion, Epididymitis |
| Arm X-Ray | ACR | Arm / Wrist / Shoulder Fracture |
| Appendix Ultrasound | ACR, NICE | Acute Appendicitis, Ectopic Pregnancy, Ovarian Torsion |

---

## Quick Start

```bash
cp .env.example .env
# Fill in at minimum: NEBIUS_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

See each subfolder's README for detailed setup and run instructions.
