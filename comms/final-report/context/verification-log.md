# Final Report Verification Log

Status: in progress  
Date: 2026-04-14

## Observations

1. Build health
The report builds successfully with `make pdf` in `comms/final-report/`.

Evidence:
- `_output/final-report.pdf` is generated and copied to `final-report.pdf`.
- `_output/final-report.log` only showed a benign caption warning during this pass:
  `Package caption Warning: Unused \captionsetup[figure] on input line 26.`

2. The manuscript is not actually final yet
There are unresolved placeholders and explicit pending-result markers in the submitted report source.

Evidence:
- `sections/06-results.tex`: `\textbf{[TODO: Alice to draft once KG validation statistics are compiled.]}`
- `sections/06-results.tex`: `GraphRAG identified 11,110 total discoveries; chart review of $n = \text{[TODO]}$ sampled cases confirmed [TODO]\% as genuinely indicated tests.`
- `sections/06-results.tex`: `\textbf{[Phase~1 MedMNIST sweep results pending. Insert table when branch \texttt{results/medmnist-sweep-v2} is available.]}`

3. The report overstates how complete the evidence base is
`sections/05-evaluation.tex` says chart review results are reported in the results section, but the corresponding results are still TODOs. The abstract and discussion should be checked carefully to avoid implying completion beyond what the manuscript actually contains.

Evidence:
- `sections/05-evaluation.tex`: `A clinical domain expert performed retrospective chart review on a random sample of newly imputed labels to validate that discoveries reflect genuine false negatives; results are reported in Section~\ref{sec:results}.`
- `sections/06-results.tex` still contains unresolved chart-review placeholders.

4. Multiple KG String rows do not match the committed full-cohort validation CSV
The committed full-cohort validation CSV for graph-only substring matching only cleanly supports the ECG row. Arm X-Ray and Appendix Ultrasound diverge substantially from the manuscript table, and Testicular Ultrasound has a smaller discovery-count mismatch.

Examples:
- Report Arm X-Ray row:
  `0.009 / 0.031 / 0.013 / 6,696`
- CSV Arm X-Ray row:
  `0.0049 / 0.0131 / 0.0072 / 5,040`
- Report Appendix Ultrasound row:
  `0.809 / 0.793 / 0.801 / 617`
- CSV Appendix Ultrasound row:
  `0.0503 / 0.7936 / 0.0946 / 49,354`
- Report Testicular Ultrasound discoveries:
  `20,035`
- CSV Testicular Ultrasound FP count:
  `20,072`

Current interpretation:
- The KG String table in the manuscript is not reproducible from the committed full-validation CSV.
- At minimum, Arm X-Ray and Appendix Ultrasound need source reconciliation before the report can be called correct.
- This remains the strongest hard correctness issue found.

5. BloodMNIST table is grounded in committed results
The BloodMNIST replication table in `sections/06-results.tex` matches `lilaw-poc/results/medmnist/results.json`.

Verified means from JSON:
- `0%`: baseline `96.39%`, LiLAW `96.74%`
- `20%`: baseline `94.47%`, LiLAW `94.13%`
- `40%`: baseline `92.94%`, LiLAW `92.78%`
- `60%`: baseline `90.15%`, LiLAW `89.62%`
- `80%`: baseline `82.09%`, LiLAW `82.59%`

6. `lilaw-poc` test suite passes in the intended dev environment
The local package tests passed after invoking the project with its `dev` extra.

Command:
- `uv run --extra dev python -m pytest -q tests`

Result:
- `58 passed in 31.49s`

7. Knowledge-graph audit statistics are not reproducible from committed artifacts yet
The report describes a four-persona audit pipeline and the results section explicitly expects counts for `Verified`, `Flagged for Review`, `Rejected`, and mean confidence, but no committed `knowledge-graphs/audit_report.json` was found in this worktree during the current pass.

Evidence:
- `knowledge-graphs/README.md` documents `audit_report.json` as a generated artifact.
- `sections/06-results.tex` expects audit summary statistics.
- No committed `knowledge-graphs/audit_report.json` has been located so far.

8. Full GraphRAG imputation metrics are not yet reproducible from committed result files
I have not yet located a committed full-cohort semantic retrieval validation CSV/JSON that backs the report’s GraphRAG rows or the aggregate `11,110` discoveries count.

Current state:
- `data-processing/student_data_enriched_graphonly_full_validation.csv` exists for the graph-only substring method.
- I have not yet found an equivalent committed full-cohort semantic-validation artifact.
- The GraphRAG table values therefore remain only partially verified at this point.

9. Training-configuration prose does not exactly match the notebook in repo
The report says training used early stopping on validation loss, but the notebook context currently shows early stopping on validation AUPRC, while `ReduceLROnPlateau` monitors validation loss.

Report:
- `sections/05-evaluation.tex`: `All conditions use identical hyperparameters to isolate the effect of label strategy: RAdam (lr $10^{-3}$, weight decay $10^{-4}$) with ReduceLROnPlateau scheduling, up to 15 epochs with early stopping on validation loss, batch size 64.`

Notebook evidence:
- `Sample_training_for_students.ipynb` defines `EarlyStopping(monitor="val_auprc", patience=3, mode="max")`
- `Sample_training_for_students.ipynb` defines `ReduceLROnPlateau(... monitor: "val_loss")`
- `Sample_training_for_students.ipynb` defines `Trainer(max_epochs=15, ...)`
- `Sample_training_for_students.ipynb` builds `DataLoader(... batch_size=64, ...)`

Current interpretation:
- `max_epochs=15`, `RAdam`, and `batch_size=64` are supported.
- The early-stopping monitor in the report appears incorrect unless another training config exists elsewhere.

10. The exact text-encoder claim is not fully backed by local context
The report names `ClinicalBERT` explicitly, but the local notebook context only confirms a BERT-via-MLflow text embedding path, not the exact encoder identity.

Evidence:
- `sections/05-evaluation.tex`: `Three free-text fields are each encoded by ClinicalBERT...`
- `Sample_training_for_students.ipynb` imports `batch_predict` from `src.mlmd.train.data.encoders.nlp_bert`
- `Sample_training_for_students.ipynb` describes the component as `BERT-via-MLflow text embedder`
- I have not yet found a second committed source in this repo that names `ClinicalBERT` directly.

11. The manuscript promises calibration reporting, but no local implementation/result artifact has been located yet
The report says ECE is reported, but I have not yet found committed evaluation code or result files in this repo that compute or store ECE for the integrated MLMD runs.

Evidence:
- `sections/05-evaluation.tex` says `Expected Calibration Error (ECE)` is reported.
- Planning/context docs repeat ECE as a target metric.
- The training notebook context I inspected logs `AUROC`, `AUPRC`, and `F1`, but I have not yet found ECE computation in the same local context.

12. The KG-construction model claim is not pinned in repo code
The report says `GPT-5-mini` generated initial triage rules, but the KG codebase appears provider-configurable and the committed code/docs I inspected do not pin that exact model name.

Evidence:
- `sections/04-methods.tex`: `GPT-5-mini generates initial triage rules...`
- `knowledge-graphs/README.md` documents configurable LLM provider settings through `.env`
- `knowledge-graphs/filter_guidelines.py` supports the EPFL guidelines dataset claim
- I have not yet found committed code or config in this worktree that fixes the generator to `GPT-5-mini`

13. Several prose passages currently convert candidate discoveries into confirmed recovered events
This is stronger than the evidence actually shown in the manuscript. The evaluation section correctly defines discoveries as candidate false negatives, and the chart-review confirmation is still left as TODO.

Examples:
- `sections/05-evaluation.tex` defines discoveries as candidate false negatives.
- `sections/06-results.tex` still leaves chart-review confirmation unresolved.
- But `sections/00-abstract.tex` says the pipeline `recovers 11,110 previously unrecorded test events`.
- `sections/07-discussion-and-limitations.tex` says `The GraphRAG pipeline recovered 11,110 test-ordering events the production regex system misclassified as negatives.`

Current interpretation:
- Until chart-review confirmation is filled in, those statements should be softened to candidate discoveries / potential false negatives.

14. No committed chart-review output artifact has been located yet
I searched for likely review/sample outputs and did not find a committed discovery-review CSV or audit report artifact that would let another reviewer reproduce the chart-review claims.

Search result:
- No committed `audit_report.json`
- No committed discovery-review sample file
- No committed full semantic-validation CSV/JSON located so far

15. The imputation latency numbers are not yet traceable to a committed benchmark artifact
The report gives precise initialization and per-encounter latency values, but I have not yet found a committed benchmark log, notebook output, or CSV/JSON that reproduces them.

Evidence:
- `sections/06-results.tex` reports:
  - `0.55 ms` per encounter, `6.16 ms` initialization for string matching
  - `175.54 ms` per encounter, `17.75 s` initialization for GraphRAG
- I have not yet found a corresponding committed benchmark artifact in this worktree.

16. One MedMNIST diagnostic sentence overstates the committed `beta` trend
The BloodMNIST table itself is correct, but one prose interpretation in `sections/06-results.tex` does not exactly match the committed JSON summary.

Report:
- `sections/06-results.tex`: `\beta and \delta did adapt monotonically with noise (increasing by 0.6 and 1.1 from 0\% to 80\%)`

Committed artifact:
- `lilaw-poc/results/medmnist/results.json`

Computed means by noise rate:
- `beta`: `2.1267, 2.4294, 2.5821, 2.6547, 2.5796`
- `delta`: `6.0544, 6.1815, 6.4407, 6.8904, 7.1992`

Current interpretation:
- `delta` does increase monotonically and by about `+1.1`.
- Mean `beta` does not increase monotonically through `80%`; it rises through `60%` and then dips slightly at `80%`.
- The report’s `beta` summary should be softened or recalculated.

17. Cohort-summary counts are not yet traceable to a committed local artifact
I have not yet found a committed CSV/JSON/notebook output in this worktree that reproduces the exclusion counts, final cohort size, or pathway prevalences stated in `sections/03-data-and-setting.tex`.

Examples from the report:
- `25,397` left-without-being-seen exclusions
- `11,396` missing-diagnosis exclusions
- final cohort `N = 483,705`
- pathway positives `1,904 / 12,236 / 3,293 / 2,889`

Current interpretation:
- These numbers may still be correct, but they are not currently reproducible from the committed artifacts I have inspected.

18. The Adult LiLAW proxy table is at least internally consistent with the local LiLAW docs
The PR-AUC and Rec@PPV80 values in `sections/06-results.tex:62-65` match the corresponding summary table in `lilaw-poc/docs/DECISIONS.md:81-84` to reported precision.

19. The KG table inconsistencies were already present in Alice's rough draft
The current imputation table and surrounding KG prose are not new to the final-report rewrite; the same values and claims already appear in `rough-drafts/conf-paper.tex`.

High-signal refs:
- `rough-drafts/conf-paper.tex:172-201` contains the same KG String and GraphRAG summary language.
- `rough-drafts/conf-paper.tex:183-193` contains the same table values now shown in `sections/06-results.tex:30-40`.

Practical implication:
- These should be surfaced back to Alice as rough-draft-era inconsistencies to review.
- No further source-tracing is necessary for the current verification pass.

20. Phase 1 MedMNIST is not part of this report's scope
For this report, the relevant issue is not that Phase 1 needs to be executed; it is that the manuscript still visibly references pending Phase 1 results.

Practical implication:
- `sections/06-results.tex:96,126` should be treated as cleanup/removal targets if Phase 1 is definitively out of scope for the report.

## Reader-Facing Triage

If the goal is to catch what a normal skeptical reader would realistically challenge, the highest-signal issues are:

1. The manuscript still contains visible non-final placeholders.
- `sections/06-results.tex:11` leaves the entire KG integrity subsection as `[TODO]`.
- `sections/06-results.tex:19` leaves chart-review counts and confirmation rate as `[TODO]`.
- `sections/06-results.tex:126` says the Phase 1 MedMNIST sweep results are still pending.

2. The abstract and discussion overclaim confirmation beyond what the manuscript shows.
- `sections/00-abstract.tex:4-6` says the pipeline is `clinically validated` and `recovered 11,110 previously unrecorded test events`.
- `sections/05-evaluation.tex:33` defines discoveries as candidate false negatives and says chart-review results appear in the results section.
- `sections/06-results.tex:19` still leaves those chart-review results as TODOs.
- `sections/07-discussion-and-limitations.tex:6` again states the 11,110 events as recovered test-ordering events, not candidates.

3. One main results table appears numerically wrong against the local validation CSV.
- `sections/06-results.tex:33,36,39` disagree with `data-processing/student_data_enriched_graphonly_full_validation.csv:3-5`.
- The strongest examples are Arm X-Ray (`0.009/0.031/0.013/6,696` in the report vs `0.0049/0.0131/0.0072/5,040` in the CSV) and Appendix Ultrasound (`0.809/0.793/0.801/617` vs `0.0503/0.7936/0.0946/49,354`).

4. The paper’s current framing is explicitly partial, not a completed end-to-end evaluation.
- `sections/06-results.tex:130` says the four-condition downstream ablation `is not complete at submission`.
- `sections/00-abstract.tex:6` is mostly consistent on this point, but the overall manuscript still reads more complete than the body evidence actually is because of the unresolved imputation-validation and KG-validation placeholders above.
