#!/usr/bin/env python3
"""Emit recommended experiment commands for the aggressive follow-up plan."""

from __future__ import annotations

import argparse
from pathlib import Path


GPU_SEEDS = "42 123 456"
CPU_SEEDS = "42 123 456 789 2024"


def gpu_phase1_jobs(results_root: str) -> list[str]:
    return [
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets bloodmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q1_blood_full"
        ),
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets dermamnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q1_derma_full"
        ),
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets pathmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q1_path_full"
        ),
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets bloodmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 30 "
            f"--output-dir {results_root}/medmnist_q2_blood_short"
        ),
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets dermamnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 30 "
            f"--output-dir {results_root}/medmnist_q2_derma_short"
        ),
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets pathmnist --noise-rates 0.0 0.3 0.5 "
            f"--seeds {GPU_SEEDS} --epochs 30 "
            f"--output-dir {results_root}/medmnist_q2_path_short"
        ),
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets bloodmnist --noise-rates 0.6 0.8 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q3_blood_highnoise"
        ),
        (
            "uv run python scripts/run_medmnist_matrix.py "
            "--datasets pathmnist --noise-rates 0.6 0.8 "
            f"--seeds {GPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/medmnist_q3_path_highnoise"
        ),
    ]


def cpu_phase1_jobs(results_root: str) -> list[str]:
    return [
        (
            "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python scripts/run_tabular_matrix.py "
            "--datasets adult --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 50 "
            f"--output-dir {results_root}/tabular_q4_adult_short"
        ),
        (
            "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python scripts/run_tabular_matrix.py "
            "--datasets adult --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 150 "
            f"--output-dir {results_root}/tabular_q4_adult_long"
        ),
        (
            "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python scripts/run_tabular_matrix.py "
            "--datasets breast_cancer --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/tabular_q5_breast_control"
        ),
        (
            "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python scripts/run_tabular_matrix.py "
            "--datasets pima --noise-rates 0.1 0.2 0.3 0.4 0.5 0.6 "
            f"--seeds {CPU_SEEDS} --epochs 100 "
            f"--output-dir {results_root}/tabular_q5_pima_control"
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=["gpu_phase1", "cpu_phase1", "all_phase1"],
        help="Job profile to emit.",
    )
    parser.add_argument(
        "--results-root",
        default="results/priority",
        help="Root directory for output directories embedded in the commands.",
    )
    parser.add_argument(
        "--output",
        help="Optional output file. If omitted, commands are printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs: list[str] = []
    if args.profile in {"gpu_phase1", "all_phase1"}:
        jobs.extend(gpu_phase1_jobs(args.results_root))
    if args.profile in {"cpu_phase1", "all_phase1"}:
        jobs.extend(cpu_phase1_jobs(args.results_root))

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
