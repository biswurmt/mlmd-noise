# Task 10: Smoke Test — End-to-End

**Files:**
- Create: `lilaw-poc/tests/test_e2e.py`

- [ ] **Step 1: Write end-to-end smoke test**

```python
# tests/test_e2e.py
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
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_e2e.py -v -m slow`
Expected: PASS. Full pipeline runs without errors.

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Final lint + type check**

Run: `ruff check src/ tests/ && ty check src/`

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(lilaw-poc): add end-to-end smoke test"
```

---

## Completion

All 10 tasks complete. The LiLAW PoC is now ready for:
1. Full experimental sweep (`python -m lilaw_poc.experiment`)
2. Paper verification checkpoint 2 (re-verify code against LiLAW paper)
3. Results analysis and publication
