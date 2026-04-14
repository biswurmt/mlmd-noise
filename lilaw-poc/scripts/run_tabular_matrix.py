#!/usr/bin/env python3
"""Wrapper around the tabular asymmetric-noise sweep with explicit CLI arguments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lilaw_poc.experiment import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Tabular dataset names, e.g. adult pima breast_cancer",
    )
    parser.add_argument(
        "--noise-rates",
        nargs="+",
        type=float,
        required=True,
        help="Noise rates as decimals, e.g. 0.1 0.2 0.3",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for results.")
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-form note stored beside the results.",
    )
    return parser.parse_args()


def write_suite_config(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_config = {
        "track": "tabular",
        "datasets": args.datasets,
        "noise_rates": args.noise_rates,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "output_dir": str(output_dir),
        "notes": args.notes,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    config_path = output_dir / "suite_config.json"
    config_path.write_text(json.dumps(suite_config, indent=2) + "\n")
    return config_path


def main() -> None:
    args = parse_args()
    config_path = write_suite_config(args)
    print(f"Wrote suite config to {config_path}")
    print(
        "Running tabular sweep: "
        f"datasets={args.datasets} noise_rates={args.noise_rates} "
        f"seeds={args.seeds} epochs={args.epochs}"
    )
    run_sweep(
        datasets=args.datasets,
        noise_rates=args.noise_rates,
        seeds=args.seeds,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
