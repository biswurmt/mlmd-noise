# Final Report

Final report source for CSC2541H:
`Reading Between the Lines: LLM-Based Label Imputation and Meta-Learning for Noisy Labels in Emergency Department Triage`

This repo is for writing and assembling the paper, not for running the ML pipeline itself.

## Start here

If you are arriving cold, read these in order:

1. `context/project-background-short.md`
2. `context/scientific-communication-guide.md`
3. `context/final-report-guidance.md`
4. `COLLABORATION.md`

That gives you the project framing, the writing rules, the required report structure, and the local editing constraints.

## What this paper is about

The paper studies asymmetric label noise in the SickKids emergency-department MLMD workflow. Positive test-order labels are reliable. Negative labels are ambiguous because some tests were performed at external facilities and never recorded in the SickKids EHR. The report evaluates two interventions:

- label imputation via an audited diagnosis-to-test knowledge graph
- robust training via LiLAW meta-learning

The core question is whether these methods improve recall, the clinical efficiency metric, without sacrificing precision, the safety metric.

## Repo map

- `final-report.tex`: main LaTeX entrypoint
- `sections/`: report content files
- `references.bib`: bibliography
- `assets/`: figures and other static report assets
- `context/`: read-only background, style, and course guidance
- `styles/`: LaTeX class and formatting support

Current section files:

- `sections/00-abstract.tex`
- `sections/01-introduction-clinical-motivation.tex`
- `sections/02-data-and-setting-methods.tex`
- `sections/03-evaluation-and-results.tex`
- `sections/04-discussion-and-limitations.tex`
- `sections/05-future-work.tex`

## Working rules

- Edit report prose in `sections/` unless there is a specific reason to touch another file.
- Treat `context/` as read-only.
- Do not invent metrics, claims, or comparisons not supported by the actual analysis.
- Keep the clinical framing explicit: recall is workflow efficiency, precision is safety.
- Preserve the decision-support framing. The system assists clinicians; it does not replace them.

## Build

```bash
make pdf
make view
make clean
```

After any material edit, rebuild and check `final-report.pdf` for:

- broken compilation
- overfull boxes or layout issues
- missing figures
- unresolved citations or references
- heading/order mismatches with the guidance

## Metrics that must stay visible

- Recall: primary clinical efficiency metric
- Precision: safety metric
- PR curves: report threshold tradeoffs, not only a single operating point
- ECE: calibration / clinician trust
- Label recovery rate: imputation effectiveness

## When editing sections

- Introduction should ground the reader in the SickKids workflow before abstract ML framing.
- Related work should identify gaps, not just summarize papers.
- Methods must describe all four experimental conditions clearly.
- Results must explain clinical meaning, not only model performance.
- Discussion must be honest about limitations and maintain the decision-support framing.
- Future work should stay concrete and short.

## Related docs

- `COLLABORATION.md`: authoritative local collaboration and writing constraints
- `AGENTS.md`: agent-facing instructions in this workspace
