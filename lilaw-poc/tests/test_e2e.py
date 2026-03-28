"""End-to-end smoke tests: single experiment and full sweep."""

import json

import pytest

from lilaw_poc.experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_single_experiment,
    run_sweep,
)


class TestEndToEnd:
    """Smoke test that the full pipeline runs without errors."""

    @pytest.mark.slow
    def test_single_experiment_completes(self) -> None:
        """A single experiment should complete and return valid metrics."""
        config = ExperimentConfig(
            dataset="breast_cancer",
            noise_rate=0.2,
            seed=42,
            epochs=5,  # short for CI
            batch_size=32,
        )
        result = run_single_experiment(config)

        assert 0.0 <= result.baseline_pr_auc <= 1.0
        assert 0.0 <= result.lilaw_pr_auc <= 1.0
        assert 0.0 <= result.baseline_recall_at_ppv80 <= 1.0
        assert 0.0 <= result.lilaw_recall_at_ppv80 <= 1.0
        assert "alpha" in result.final_meta_params

    @pytest.mark.slow
    def test_run_sweep_writes_json_and_returns_results(self, tmp_path: object) -> None:
        """run_sweep should write results.json and return ExperimentResult list."""
        out_dir = str(tmp_path)
        results = run_sweep(
            datasets=["breast_cancer"],
            noise_rates=[0.2],
            seeds=[42],
            epochs=3,
            output_dir=out_dir,
        )

        # Returns correct type and count
        assert len(results) == 1
        assert isinstance(results[0], ExperimentResult)

        # JSON file written with expected structure
        json_path = tmp_path / "results.json"  # type: ignore[operator]
        assert json_path.exists()  # type: ignore[union-attr]
        with open(json_path) as f:  # type: ignore[arg-type]
            data = json.load(f)
        assert len(data) == 1
        entry = data[0]
        assert entry["dataset"] == "breast_cancer"
        assert entry["noise_rate"] == 0.2
        assert entry["seed"] == 42
        assert "baseline_pr_auc" in entry
        assert "lilaw_pr_auc" in entry
        assert "final_meta_params" in entry
