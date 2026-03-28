import numpy as np
import torch

from lilaw_poc.data.noise import inject_asymmetric_noise


class TestInjectAsymmetricNoise:
    """Test asymmetric positive→negative label noise injection."""

    def test_only_positive_labels_flipped(self) -> None:
        """No negative label should ever become positive."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=0.5, rng=rng)
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
        assert (noisy[neg_indices] == 0).all(), "Negative labels must never be flipped"

    def test_flip_rate_zero_no_change(self) -> None:
        """0% flip rate should return identical labels."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 1, 1, 0, 1])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=0.0, rng=rng)
        assert torch.equal(noisy, labels)
        assert mask.sum() == 0

    def test_flip_rate_one_flips_all_positives(self) -> None:
        """100% flip rate should flip every positive to negative."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 1, 1, 0, 1])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=1.0, rng=rng)
        assert (noisy == 0).all()
        assert mask.sum() == 3  # 3 positives flipped

    def test_approximate_flip_rate(self) -> None:
        """Over many samples, actual flip rate should approximate target."""
        rng = np.random.default_rng(42)
        labels = torch.ones(10000, dtype=torch.long)
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=0.3, rng=rng)
        actual_rate = mask.sum().item() / labels.sum().item()
        assert abs(actual_rate - 0.3) < 0.03, f"Expected ~0.3, got {actual_rate}"

    def test_returns_flip_mask(self) -> None:
        """Mask should be True exactly where labels were flipped."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 1, 1, 0, 1, 1, 0])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=1.0, rng=rng)
        expected_mask = torch.tensor([False, True, True, False, True, True, False])
        assert torch.equal(mask, expected_mask)
