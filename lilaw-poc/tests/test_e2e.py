"""End-to-end smoke test: run a single experiment on breast cancer with 20% noise."""

import pytest

from lilaw_poc.experiment import ExperimentConfig, run_single_experiment


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
