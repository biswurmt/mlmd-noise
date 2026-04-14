"""Tests for job generation used in smoke and phase-1 dispatch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_job_builder(profile: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "scripts/build_priority_jobs.py", "--profile", profile],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_gpu_phase1_jobs_use_venv_python() -> None:
    jobs = _run_job_builder("gpu_phase1")

    assert jobs
    assert all(
        job.startswith("./.venv/bin/python -u scripts/run_medmnist_matrix.py") for job in jobs
    )
    assert "uv run python" not in "\n".join(jobs)


def test_gpu_smoke_jobs_default_to_smoke_results_root() -> None:
    jobs = _run_job_builder("gpu_smoke")

    assert len(jobs) == 3
    assert all("results/smoke/" in job for job in jobs)
    assert all("--epochs 2" in job for job in jobs)
