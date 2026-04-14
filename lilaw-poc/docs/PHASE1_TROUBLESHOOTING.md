# Phase-1 Troubleshooting Algorithm

This runbook is for a collaborator or smaller agent handling `lilaw-poc` phase-1 experiments on the remote GPU host.

The objective is not to guess. The objective is to classify the failure mode quickly, collect the right artifact, and either recover or stop cleanly.

## Inputs

- Repo root: `lilaw-poc/`
- Virtualenv interpreter: `./.venv/bin/python`
- GPU dispatcher: `scripts/dispatch_queue.sh`
- Monitor: `scripts/monitor_dispatch.sh`
- Job builder: `scripts/build_priority_jobs.py`

## Algorithm

1. Confirm you are in the right directory.
   - Run: `cd lilaw-poc && pwd`
   - Expected: working directory is the project root that contains `pyproject.toml`, `scripts/`, `src/`, `jobs/`, and `results/`.

2. Confirm the venv exists and the launcher path is valid.
   - Run: `test -x ./.venv/bin/python && ./.venv/bin/python --version`
   - If this fails, stop and recreate the env with `./scripts/setup_env.sh --with-medmnist`.

3. Confirm PyTorch sees CUDA on the GPU host.
   - Run:
     ```bash
     ./.venv/bin/python - <<'PY'
     import torch
     print("torch", torch.__version__)
     print("cuda_available", torch.cuda.is_available())
     print("device_count", torch.cuda.device_count())
     PY
     ```
   - If `cuda_available` is `False` on the GPU host, stop. Do not debug phase-1 jobs before fixing the environment.

4. Regenerate job files from the repo, do not trust stale copies.
   - GPU phase 1:
     `./.venv/bin/python scripts/build_priority_jobs.py --profile gpu_phase1 --output jobs/gpu_phase1.txt`
   - CPU phase 1:
     `./.venv/bin/python scripts/build_priority_jobs.py --profile cpu_phase1 --output jobs/cpu_phase1.txt`
   - Smoke jobs:
     `./.venv/bin/python scripts/build_priority_jobs.py --profile gpu_smoke --output jobs/gpu_smoke.txt`
     `./.venv/bin/python scripts/build_priority_jobs.py --profile cpu_smoke --output jobs/cpu_smoke.txt`

5. Validate the generated job files before launch.
   - Run: `rg -n "uv run python" jobs/*.txt`
   - Expected: no matches.
   - Run: `sed -n '1,20p' jobs/gpu_phase1.txt`
   - Expected: commands start with `./.venv/bin/python -u`.

6. Run smoke first if phase-1 has not completed cleanly on the current machine state.
   - GPU smoke:
     `bash scripts/dispatch_queue.sh --job-file jobs/gpu_smoke.txt --gpus 0,1,2 --log-dir logs/gpu_smoke`
   - CPU smoke:
     `bash scripts/dispatch_queue.sh --job-file jobs/cpu_smoke.txt --max-parallel 1 --log-dir logs/cpu_smoke`
   - Monitor:
     `bash scripts/monitor_dispatch.sh --log-dir logs/gpu_smoke --results-dir results/smoke`

7. If smoke passes, launch phase 1 and start the monitor in a separate shell.
   - GPU:
     `bash scripts/dispatch_queue.sh --job-file jobs/gpu_phase1.txt --gpus 0,1,2,3,4,5,6,7 --log-dir logs/gpu_phase1`
   - CPU:
     `bash scripts/dispatch_queue.sh --job-file jobs/cpu_phase1.txt --max-parallel 4 --log-dir logs/cpu_phase1`
   - Monitor:
     `bash scripts/monitor_dispatch.sh --log-dir logs/gpu_phase1 --results-dir results/priority`

8. For any suspect job, inspect three artifacts in this order.
   - Status file: `logs/<queue>/job_XX.status`
   - Dispatcher log: `logs/<queue>/job_XX.log`
   - Suite log: `results/.../run.log`

9. Classify the job using the status file.
   - `state=running`: process was launched and has not exited yet.
   - `state=completed`: queue layer succeeded; inspect suite outputs if results are missing.
   - `state=failed`: command exited non-zero; the traceback or shell error should be in `job_XX.log`.

10. Apply the decision rules below.

## Decision Rules

### A. Job file still contains `uv run python`

- Cause: stale or manually edited jobs.
- Action:
  - regenerate with `scripts/build_priority_jobs.py`
  - replace the stale job file
  - relaunch only the failed jobs

### B. `job_XX.log` has only the launch header and no training output yet

- Likely cause: cold start.
  - first-time dataset download
  - Python import latency
  - native library initialization
- Action:
  - check whether `job_XX.status` still says `state=running`
  - check `nvidia-smi`
  - wait up to 10 minutes before calling it stuck
  - check for the suite directory and `suite_config.json`

### C. `job_XX.status` says `failed`

- Action:
  - read the traceback in `job_XX.log`
  - rerun the exact command from `job_XX.cmd` manually
  - only patch code after you can reproduce the failure directly

### D. GPU job is running but `nvidia-smi` shows 0% utilization for more than 10 minutes

- Action:
  - inspect `results/.../run.log`
  - if no new lines are appearing, inspect `job_XX.log`
  - check for dataset download stalls or repeated restarts
  - if process is alive but progress is frozen, kill and rerun the single command manually

### E. Tabular job crashes with tensor/device mismatch

- Expected fix already exists in `src/lilaw_poc/experiment.py`.
- Action:
  - confirm the local code contains inference on the model device and returns CPU scores
  - rerun the single tabular command manually
  - if the traceback differs, debug that new failure rather than assuming it is the same bug

### F. `results.json` is missing but the dispatcher says completed

- Action:
  - inspect the suite `run.log`
  - confirm the output directory in `suite_config.json`
  - verify the command in `job_XX.cmd` points at the intended `--output-dir`
  - rerun manually if the suite exited before JSON writeout

## Manual Reproduction Rule

When a queued job fails, reproduce with the exact command from `job_XX.cmd`.

Example:

```bash
bash -lc "$(cat logs/gpu_phase1/job_03.cmd)"
```

Do not rewrite the command by hand unless quoting is visibly wrong.

## Recovery Rules

1. If the failure is environment-level, fix the environment first.
2. If the failure is command-generation-level, regenerate job files.
3. If the failure is code-level, patch the smallest reproducible bug and rerun smoke before phase 1.
4. If only one suite fails, rerun only that suite. Do not restart the entire batch by default.

## Acceptance Criteria

Phase-1 execution is healthy when all of the following are true:

- generated jobs use `./.venv/bin/python -u`
- dispatcher produces `job_XX.log`, `job_XX.status`, and `job_XX.cmd`
- smoke jobs complete and write `results.json`
- phase-1 jobs show either active GPU utilization or advancing `run.log` output
- completed suites write `suite_config.json`, `run.log`, and `results.json`
