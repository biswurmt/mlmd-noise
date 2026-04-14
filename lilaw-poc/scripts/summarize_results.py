#!/usr/bin/env python3
"""Summarize one or more result directories produced by the follow-up sweeps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def collect_result_files(paths: list[str]) -> list[Path]:
    result_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.name == "results.json":
            result_files.append(path)
            continue
        if path.is_dir():
            result_files.extend(sorted(path.rglob("results.json")))
    return result_files


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {pstdev(values):.4f}"


def summarize_medmnist(path: Path, rows: list[dict]) -> None:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["noise_rate"])].append(row)

    print(f"\n== {path.parent} [medmnist] ==")
    print(
        f"{'dataset':<12} {'noise':>6} {'n':>3} "
        f"{'baseline_acc':>18} {'lilaw_acc':>18} {'delta':>10}"
    )
    for (dataset, noise_rate), group in sorted(groups.items()):
        bl = [row["baseline_acc"] for row in group]
        lw = [row["lilaw_acc"] for row in group]
        deltas = [row["lilaw_acc"] - row["baseline_acc"] for row in group]
        print(
            f"{dataset:<12} {noise_rate:>6.0%} {len(group):>3} "
            f"{format_mean_std(bl):>18} {format_mean_std(lw):>18} {format_mean_std(deltas):>10}"
        )


def summarize_tabular(path: Path, rows: list[dict]) -> None:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["noise_rate"])].append(row)

    print(f"\n== {path.parent} [tabular] ==")
    print(
        f"{'dataset':<16} {'noise':>6} {'n':>3} "
        f"{'baseline_pr_auc':>18} {'lilaw_pr_auc':>18} {'delta_pr':>10} "
        f"{'baseline_rec80':>18} {'lilaw_rec80':>18}"
    )
    for (dataset, noise_rate), group in sorted(groups.items()):
        bl_auc = [row["baseline_pr_auc"] for row in group]
        lw_auc = [row["lilaw_pr_auc"] for row in group]
        delta_auc = [row["lilaw_pr_auc"] - row["baseline_pr_auc"] for row in group]
        bl_rec = [row["baseline_recall_at_ppv80"] for row in group]
        lw_rec = [row["lilaw_recall_at_ppv80"] for row in group]
        print(
            f"{dataset:<16} {noise_rate:>6.0%} {len(group):>3} "
            f"{format_mean_std(bl_auc):>18} {format_mean_std(lw_auc):>18} {format_mean_std(delta_auc):>10} "
            f"{format_mean_std(bl_rec):>18} {format_mean_std(lw_rec):>18}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result directories or results.json files to summarize.",
    )
    args = parser.parse_args()

    result_files = collect_result_files(args.paths)
    if not result_files:
        raise SystemExit("No results.json files found.")

    for result_file in result_files:
        rows = json.loads(result_file.read_text())
        if not rows:
            print(f"\n== {result_file.parent} ==\n(empty results)")
            continue
        sample = rows[0]
        if "baseline_acc" in sample:
            summarize_medmnist(result_file, rows)
        elif "baseline_pr_auc" in sample:
            summarize_tabular(result_file, rows)
        else:
            print(f"\n== {result_file.parent} ==\n(unrecognized result schema)")


if __name__ == "__main__":
    main()
