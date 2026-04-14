#!/usr/bin/env python3
"""Wrapper around the MedMNIST sweep with explicit CLI arguments."""

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

from lilaw_poc.medmnist.experiment import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True, help="MedMNIST dataset names.")
    parser.add_argument(
        "--noise-rates",
        nargs="+",
        type=float,
        required=True,
        help="Noise rates as decimals, e.g. 0.0 0.3 0.5",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for results.")
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-form note stored beside the results.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on: auto (cuda if available), cuda, or cpu.",
    )
    return parser.parse_args()


def write_suite_config(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_config = {
        "track": "medmnist",
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

    # Prepare logging: tee stdout/stderr to a logfile under the output dir
    log_path = Path(args.output_dir) / "run.log"

    class _Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    logfile = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, logfile)
    sys.stderr = _Tee(sys.stderr, logfile)

    print(f"Wrote suite config to {config_path}")
    print(
        "Running MedMNIST sweep: "
        f"datasets={args.datasets} noise_rates={args.noise_rates} "
        f"seeds={args.seeds} epochs={args.epochs}"
    )

    # Resolve device
    import torch as _torch

    if args.device == "auto":
        dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not _torch.cuda.is_available():
            print("Requested device 'cuda' but CUDA is not available; falling back to CPU")
            dev = _torch.device("cpu")
        else:
            dev = _torch.device("cuda")
    else:
        dev = _torch.device("cpu")

    print(f"Using device: {dev}")

    run_sweep(
        datasets=args.datasets,
        noise_rates=args.noise_rates,
        seeds=args.seeds,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=dev,
    )


if __name__ == "__main__":
    main()
