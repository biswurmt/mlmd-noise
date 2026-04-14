# Aggressive Follow-Up Experiment Plan

This document is written for a collaborator who has decent Linux/DevOps instincts but little or no project-specific context.

The goal is not to produce the cleanest possible paper replication. The goal is to produce a broad, high-signal body of results quickly enough to answer a few useful questions about whether LiLAW is worth carrying into MLMD.

## Mission

Figure out whether, and in what regimes, **LiLAW is likely to improve MLMD**.

In practice that means:

1. Establish whether LiLAW can show gains on at least one paper-adjacent image benchmark when training is allowed to finish.
2. Establish whether LiLAW produces small but consistent gains under **asymmetric false-negative noise**, which is the closest proxy to the MLMD problem.
3. Decide whether the evidence is strong enough to justify a direct MLMD integration run.

## Constraints

- Time is short. Prefer breadth over exhaustive confidence intervals.
- Available hardware: `8 x RTX 2080 Ti`, `~250 GiB RAM`, strong CPU, ample disk.
- `lilaw-poc` is the fast experimentation sandbox.
- `lilaw/` contains a drop-in LiLAW patch for the real MLMD training system.
- The real MLMD baseline is not fully runnable from this repo alone; it still depends on internal assets and environment pieces documented in [../../lilaw/MLMD_TRAINING_GUIDE.md](../../lilaw/MLMD_TRAINING_GUIDE.md).

## Existing Evidence

- The tabular PoC already showed modest gains on `adult` and weak or negative results on smaller/easier datasets.
- The initial MedMNIST BloodMNIST sweep did **not** reproduce the paper's gains.
- The most plausible unresolved explanation is training budget: the early-stopped runs ended before the scheduled LR drops mattered.
- The current MedMNIST code already uses `early_stopping_patience=100`, which effectively disables the v1 failure mode.

## What To Optimize For

Run experiments that suggest answers to these questions:

### Q1. Does LiLAW help on paper-adjacent image benchmarks if we let training run long enough?

If yes, that strengthens the claim that the PoC implementation is basically sound and that the weak tabular effect is about problem structure, not a broken method.

### Q2. Is any LiLAW benefit concentrated in hard regimes rather than everywhere?

If yes, that is useful even if average gains are small. It would suggest LiLAW is a targeted robustness tool, not a general free lunch.

### Q3. On adult-like asymmetric noise, is the effect small-but-stable or brittle?

This is the best cheap proxy for MLMD, because the underlying MLMD motivation is asymmetric false negatives from missing external tests.

### Q4. Is there enough evidence to justify running LiLAW directly inside MLMD?

The bar here is not "proof." The bar is "enough signal to justify one or two controlled MLMD runs."

## Repository Surface You Need

### `lilaw-poc/`

- `src/lilaw_poc/experiment.py`: tabular asymmetric-noise sweep
- `src/lilaw_poc/medmnist/experiment.py`: MedMNIST sweep
- `scripts/run_tabular_matrix.py`: wrapper added for scripted tabular runs
- `scripts/run_medmnist_matrix.py`: wrapper added for scripted MedMNIST runs
- `scripts/dispatch_queue.sh`: simple queue runner for CPU or GPU job files
- `scripts/monitor_dispatch.sh`: periodic snapshot monitor for queue runs
- `scripts/build_priority_jobs.py`: emits recommended phase-1 command files
- `scripts/summarize_results.py`: prints grouped summaries from result directories

### `lilaw/`

- `README.md`: how to swap LiLAW into the MLMD Lightning module
- `lilaw_module.py`: actual MLMD LiLAW patch
- `MLMD_TRAINING_GUIDE.md`: the least-bad orientation doc for the internal training system

## Environment Setup

Run this from the `lilaw-poc/` directory on the remote machine.

```bash
cd lilaw-poc
./scripts/setup_env.sh --with-medmnist
source .venv/bin/activate
```

Notes:

- Use `./.venv/bin/python -u` for long-running experiment commands and generated job files.
- MedMNIST and OpenML datasets will download on first use. Keep caches on local disk.
- Use `tmux` or `screen` for all long-running launch sessions.
- Use [PHASE1_TROUBLESHOOTING.md](./PHASE1_TROUBLESHOOTING.md) as the operational runbook if jobs stall or fail.

## Phase 0: Smoke Test

Do this before flooding the machine with jobs.

### 0A. Generate smoke job files

```bash
mkdir -p jobs logs results/smoke

./.venv/bin/python scripts/build_priority_jobs.py \
  --profile gpu_smoke \
  --output jobs/gpu_smoke.txt

./.venv/bin/python scripts/build_priority_jobs.py \
  --profile cpu_smoke \
  --output jobs/cpu_smoke.txt
```

### 0B. Launch smoke jobs

```bash
bash scripts/dispatch_queue.sh \
  --job-file jobs/gpu_smoke.txt \
  --gpus 0,1,2 \
  --log-dir logs/gpu_smoke

bash scripts/dispatch_queue.sh \
  --job-file jobs/cpu_smoke.txt \
  --max-parallel 1 \
  --log-dir logs/cpu_smoke
```

### 0C. Monitor smoke jobs

```bash
bash scripts/monitor_dispatch.sh \
  --log-dir logs/gpu_smoke \
  --results-dir results/smoke
```

Success criteria:

- both jobs complete
- MedMNIST dataset download works
- OpenML fetch works
- `results.json` is written in each output directory

If smoke fails, fix that before doing anything else.

## Phase 1: Priority Breadth-First Sweep

This is the main deliverable. Run this first.

### Why these suites

The priority suites are intentionally designed to answer several questions at once:

- long-vs-short training budget on MedMNIST
- moderate vs high noise on MedMNIST
- adult-like asymmetric noise stability on tabular data
- control behavior on small/easy tabular datasets

### Generate the recommended job files

```bash
mkdir -p jobs logs results/priority

./.venv/bin/python scripts/build_priority_jobs.py \
  --profile gpu_phase1 \
  --output jobs/gpu_phase1.txt

./.venv/bin/python scripts/build_priority_jobs.py \
  --profile cpu_phase1 \
  --output jobs/cpu_phase1.txt
```

### Launch the GPU suites

One process per GPU. These are all MedMNIST runs.

```bash
bash scripts/dispatch_queue.sh \
  --job-file jobs/gpu_phase1.txt \
  --gpus 0,1,2,3,4,5,6,7 \
  --log-dir logs/gpu_phase1
```

### Launch the CPU suites

The tabular code is CPU-only. Run it separately; do not waste GPUs on it.

```bash
bash scripts/dispatch_queue.sh \
  --job-file jobs/cpu_phase1.txt \
  --max-parallel 4 \
  --log-dir logs/cpu_phase1
```

### Monitor phase 1

Use a separate shell or `tmux` pane:

```bash
bash scripts/monitor_dispatch.sh \
  --log-dir logs/gpu_phase1 \
  --results-dir results/priority
```

For CPU runs:

```bash
bash scripts/monitor_dispatch.sh \
  --log-dir logs/cpu_phase1 \
  --results-dir results/priority \
  --process-pattern 'run_tabular_matrix.py'
```

### What the phase-1 jobs actually do

#### GPU phase 1

| Suite | Question | Settings |
|---|---|---|
| `medmnist_q1_blood_full` | Does full training unlock LiLAW? | BloodMNIST, noise `0/30/50%`, seeds `42/123/456`, `100` epochs |
| `medmnist_q1_derma_full` | same | DermaMNIST, same grid |
| `medmnist_q1_path_full` | same | PathMNIST, same grid |
| `medmnist_q2_blood_short` | Was v1 mostly a budget issue? | BloodMNIST, same grid, `30` epochs |
| `medmnist_q2_derma_short` | same | DermaMNIST, same grid, `30` epochs |
| `medmnist_q2_path_short` | same | PathMNIST, same grid, `30` epochs |
| `medmnist_q3_blood_highnoise` | Does signal appear only when noise is extreme? | BloodMNIST, noise `60/80%`, seeds `42/123/456`, `100` epochs |
| `medmnist_q3_path_highnoise` | same | PathMNIST, same grid |

#### CPU phase 1

| Suite | Question | Settings |
|---|---|---|
| `tabular_q4_adult_short` | Is the adult proxy lift visible at all? | Adult, noise `10-60%`, five seeds, `50` epochs |
| `tabular_q4_adult_long` | Does more training change the adult story? | Adult, same grid, `150` epochs |
| `tabular_q5_breast_control` | Is LiLAW mostly irrelevant on easy data? | Breast Cancer, noise `10-60%`, five seeds, `100` epochs |
| `tabular_q5_pima_control` | Does small-data instability persist? | Pima, same grid, `100` epochs |

## Phase 1 Result Triage

Do **not** wait for every last run if the story becomes obvious early.

Summarize everything written under `results/priority/`:

```bash
./.venv/bin/python scripts/summarize_results.py results/priority
```

### Decision rules

Treat the following as meaningful signals:

- MedMNIST: LiLAW improves mean accuracy by about `>= 1.0` percentage point on at least one dataset at `30-50%` noise in the `100`-epoch runs.
- MedMNIST budget sensitivity: the `100`-epoch runs outperform the `30`-epoch runs more for LiLAW than for baseline.
- Adult proxy: LiLAW improves PR-AUC or Recall@PPV floor across several adjacent noise rates, even if each gain is small.
- Control datasets: Breast Cancer stays flat and Pima stays unstable. That is fine; it helps define the boundary conditions.

### If the results are flat or negative

That is still useful. The fallback story is:

- LiLAW does not appear to give broad, paper-scale gains.
- Any benefit is likely narrow, data-regime-specific, or weaker under asymmetric tabular noise than the paper suggests.
- Direct MLMD integration should be justified only if the Adult proxy and at least one image benchmark still show a non-negative trend.

## Phase 2: Narrow Follow-Up Only Where Signal Exists

Only run this if phase 1 shows something worth sharpening.

### If a MedMNIST dataset looks promising

Promote that dataset to:

- noise `0/20/40/60/80%`
- seeds `42/123/456/789/2024`
- `100` epochs

Command pattern:

```bash
./.venv/bin/python -u scripts/run_medmnist_matrix.py \
  --datasets <promising_dataset> \
  --noise-rates 0.0 0.2 0.4 0.6 0.8 \
  --seeds 42 123 456 789 2024 \
  --epochs 100 \
  --output-dir results/followup/<promising_dataset>_fullgrid
```

### If Adult looks promising

Promote Adult only. Do **not** spend time deepening Breast Cancer or Pima unless they unexpectedly become interesting.

Suggested follow-up:

- Adult only
- noise `10/20/30/40/50/60%`
- seeds `42/123/456/789/2024`
- compare `50` vs `150` epochs

## Phase 3: MLMD Bridge Run

This phase is optional and only makes sense if the collaborator has access to the real MLMD training environment and missing internal assets.

### Why this phase exists

The actual decision we care about is not "does LiLAW beat CE on MedMNIST?" It is "does LiLAW help the MLMD system enough to be worth integrating?"

### Minimum useful MLMD run

Do **not** attempt all five modalities first.

Run:

1. ECG baseline, `3` seeds
2. ECG + LiLAW patch, `3` seeds
3. If ECG is neutral-to-positive, one second modality with more label sparsity, likely an ultrasound modality

### Why ECG first

- It is the only modality with explicit baseline metrics documented in the notebook artifacts.
- It is the least ambiguous place to validate that the MLMD training path is working.
- It reduces the chance of burning time debugging a rare-modality pipeline first.

### MLMD patch surface

Use the files in `../lilaw/`:

- [../../lilaw/README.md](../../lilaw/README.md)
- [../../lilaw/lilaw_module.py](../../lilaw/lilaw_module.py)

Operationally, the change is:

- replace `FusionLightningModule` with `LiLAWFusionLightningModule`
- keep the rest of the training loop the same

### MLMD evaluation priorities

For MLMD, the writeup should prioritize:

- PR AUC
- recall at a clinically acceptable precision floor
- absolute precision degradation, if any
- test and `tech_val` splits separately if available

If LiLAW improves recall while preserving precision bounds, that is the strongest practical argument for adoption.

## What To Write Up Quickly

If time runs out, the fastest credible writeup structure is:

1. **Implementation sanity check**
   - LiLAW was re-tested on MedMNIST with full training budget.
2. **Stress-test summary**
   - Gains were concentrated in these regimes; flat or negative in these others.
3. **MLMD relevance**
   - The adult asymmetric-noise proxy suggests LiLAW is or is not worth trying inside MLMD.
4. **Next action**
   - either patch ECG in MLMD, or stop and avoid integration cost.

## Hard Stop Conditions

Stop spending GPU time on a branch of the plan if:

- the smoke test is flaky and needs real debugging
- all three `100`-epoch MedMNIST suites are flat or negative
- Adult is flat/negative across nearly all noise rates
- the collaborator cannot access the internal MLMD assets needed to run the real system

At that point the right output is a bounded negative result, not more compute.

## Deliverables Checklist

By the end of the run window, try to leave behind:

- `results/priority/.../results.json` for every finished suite
- `suite_config.json` beside each `results.json`
- `logs/gpu_phase1/` and `logs/cpu_phase1/`
- one markdown note summarizing:
  - which suites completed
  - which questions got a tentative answer
  - whether MLMD integration is justified now
