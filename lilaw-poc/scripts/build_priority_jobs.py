#!/usr/bin/env python3
"""Emit recommended experiment commands for the aggressive follow-up plan."""

from __future__ import annotations

import argparse
from pathlib import Path


GPU_SEEDS = "42 123 456"
CPU_SEEDS = "42 123 456 789 2024"
PYTHON_CMD = "./.venv/bin/python -u"
CPU_ENV_PREFIX = "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"


def _python_job(python_cmd: str, script: str, args: str, prefix: str = "") -> str:
    prefix_text = f"{prefix} " if prefix else ""
    return f"{prefix_text}{python_cmd} {script} {args}"


def gpu_smoke_jobs(results_root: str, python_cmd: str) -> list[str]:
    return [
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets bloodmnist --noise-rates 0.0 --seeds 42 --epochs 2 "
            f"--output-dir {results_root}/medmnist_blood_e2_gpusmoke",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets dermamnist --noise-rates 0.0 --seeds 42 --epochs 2 "
            f"--output-dir {results_root}/medmnist_derma_e2_gpusmoke",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets pathmnist --noise-rates 0.0 --seeds 42 --epochs 2 "
            f"--output-dir {results_root}/medmnist_path_e2_gpusmoke",
        ),
    ]


def cpu_smoke_jobs(results_root: str, python_cmd: str) -> list[str]:
    return [
        _python_job(
            python_cmd,
            "scripts/run_tabular_matrix.py",
            "--datasets adult --noise-rates 0.2 --seeds 42 --epochs 2 "
            f"--output-dir {results_root}/tabular_adult_e2_cpusmoke",
            prefix=CPU_ENV_PREFIX,
        )
    ]


def gpu_phase1_jobs(results_root: str, python_cmd: str) -> list[str]:
    return [
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets bloodmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q1_blood_full",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets dermamnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q1_derma_full",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets pathmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q1_path_full",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets bloodmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 30 "
            f"--output-dir {results_root}/medmnist_q2_blood_short",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets dermamnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 30 "
            f"--output-dir {results_root}/medmnist_q2_derma_short",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets pathmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 30 "
            f"--output-dir {results_root}/medmnist_q2_path_short",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets bloodmnist --noise-rates 0.6 0.8 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q3_blood_highnoise",
        ),
        _python_job(
            python_cmd,
            "scripts/run_medmnist_matrix.py",
            "--datasets pathmnist --noise-rates 0.6 0.8 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q3_path_highnoise",
        ),
    ]


def cpu_phase1_jobs(results_root: str, python_cmd: str) -> list[str]:
    return [
        _python_job(
            python_cmd,
            "scripts/run_tabular_matrix.py",
            "--datasets adult --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 50 "
            f"--output-dir {results_root}/tabular_q4_adult_short",
            prefix=CPU_ENV_PREFIX,
        ),
        _python_job(
            python_cmd,
            "scripts/run_tabular_matrix.py",
            "--datasets adult --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 150 "
            f"--output-dir {results_root}/tabular_q4_adult_long",
            prefix=CPU_ENV_PREFIX,
        ),
        _python_job(
            python_cmd,
            "scripts/run_tabular_matrix.py",
            "--datasets breast_cancer --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/tabular_q5_breast_control",
            prefix=CPU_ENV_PREFIX,
        ),
        _python_job(
            python_cmd,
            "scripts/run_tabular_matrix.py",
            "--datasets pima --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/tabular_q5_pima_control",
            prefix=CPU_ENV_PREFIX,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=[
            "gpu_smoke",
            "cpu_smoke",
            "all_smoke",
            "gpu_phase1",
            "cpu_phase1",
            "all_phase1",
        ],
        help="Job profile to emit.",
    )
    parser.add_argument(
        "--results-root",
        help="Root directory for output directories embedded in the commands.",
    )
    parser.add_argument(
        "--output",
        help="Optional output file. If omitted, commands are printed to stdout.",
    )
    parser.add_argument(
        "--python-cmd",
        default=PYTHON_CMD,
        help="Python launcher prefix embedded into emitted jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    if results_root is None:
        results_root = "results/smoke" if "smoke" in args.profile else "results/priority"

    jobs: list[str] = []
    if args.profile in {"gpu_smoke", "all_smoke"}:
        jobs.extend(gpu_smoke_jobs(results_root, args.python_cmd))
    if args.profile in {"cpu_smoke", "all_smoke"}:
        jobs.extend(cpu_smoke_jobs(results_root, args.python_cmd))
    if args.profile in {"gpu_phase1", "all_phase1"}:
        jobs.extend(gpu_phase1_jobs(results_root, args.python_cmd))
    if args.profile in {"cpu_phase1", "all_phase1"}:
        jobs.extend(cpu_phase1_jobs(results_root, args.python_cmd))

    text = "\n".join(jobs) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        print(f"Wrote {len(jobs)} jobs to {output_path}")
        return

    print(text, end="")


if __name__ == "__main__":
    main()
