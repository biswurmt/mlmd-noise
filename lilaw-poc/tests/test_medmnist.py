"""Tests for the MedMNIST replication experiment modules."""

import numpy as np
import torch
import torch.nn.functional as func

from lilaw_poc.lilaw import LiLAWWeighter, compute_lilaw_weights
from lilaw_poc.medmnist.evaluate import compute_accuracy, compute_auroc
from lilaw_poc.medmnist.noise import inject_symmetric_noise

# ── Symmetric noise ──────────────────────────────────────────────────────────


class TestSymmetricNoise:
    def test_zero_noise_no_changes(self) -> None:
        labels = torch.tensor([0, 1, 2, 3, 4])
        rng = np.random.default_rng(42)
        noisy, mask = inject_symmetric_noise(labels, 0.0, 5, rng)
        assert torch.equal(noisy, labels)
        assert not mask.any()

    def test_full_noise_all_drawn(self) -> None:
        """At 100% noise rate, every label is replaced (but some may land on original)."""
        labels = torch.zeros(1000, dtype=torch.long)
        rng = np.random.default_rng(42)
        noisy, mask = inject_symmetric_noise(labels, 1.0, 10, rng)
        # With K=10, expected effective flip rate = 1.0 * 9/10 = 90%
        flip_rate = mask.float().mean().item()
        assert 0.85 < flip_rate < 0.95

    def test_noise_rate_approx(self) -> None:
        """Effective flip rate should be approximately noise_rate * (K-1)/K."""
        labels = torch.randint(0, 5, (10000,))
        rng = np.random.default_rng(42)
        noisy, mask = inject_symmetric_noise(labels, 0.4, 5, rng)
        # Expected effective rate: 0.4 * 4/5 = 0.32
        effective = mask.float().mean().item()
        assert 0.28 < effective < 0.36

    def test_labels_stay_in_range(self) -> None:
        labels = torch.randint(0, 8, (5000,))
        rng = np.random.default_rng(42)
        noisy, _ = inject_symmetric_noise(labels, 0.5, 8, rng)
        assert noisy.min() >= 0
        assert noisy.max() < 8

    def test_reproducible(self) -> None:
        labels = torch.randint(0, 5, (1000,))
        noisy1, _ = inject_symmetric_noise(labels, 0.3, 5, np.random.default_rng(99))
        noisy2, _ = inject_symmetric_noise(labels, 0.3, 5, np.random.default_rng(99))
        assert torch.equal(noisy1, noisy2)


# ── Multi-class s_label / s_max computation ──────────────────────────────────


class TestMultiClassLiLAW:
    def test_s_label_s_max_perfect_prediction(self) -> None:
        """When model is perfectly confident, s_label == s_max == 1.0."""
        logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        labels = torch.tensor([0, 1])
        probs = func.softmax(logits, dim=1)
        s_label = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        s_max = probs.max(dim=1).values
        assert torch.allclose(s_label, s_max, atol=1e-4)
        assert (s_label > 0.99).all()

    def test_s_label_s_max_wrong_prediction(self) -> None:
        """When model is confident about wrong class, s_label < s_max."""
        logits = torch.tensor([[10.0, -10.0, -10.0]])
        labels = torch.tensor([1])  # Model says class 0, but label is 1
        probs = func.softmax(logits, dim=1)
        s_label = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        s_max = probs.max(dim=1).values
        assert s_label[0] < 0.01
        assert s_max[0] > 0.99

    def test_weights_bounded(self) -> None:
        """All weight components should be in [0, 1] or reasonable range."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=1, keepdim=True)
        labels = torch.randint(0, 10, (100,))
        s_label = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        s_max = probs.max(dim=1).values

        w_a, w_b, w_d, w_total = compute_lilaw_weights(s_label, s_max, 10.0, 2.0, 6.0)
        assert (w_a >= 0).all() and (w_a <= 1).all()
        assert (w_b >= 0).all() and (w_b <= 1).all()
        assert (w_d >= 0).all() and (w_d <= 1).all()
        # w_total = w_a + w_b + w_d, so in [0, 3]
        assert (w_total >= 0).all() and (w_total <= 3).all()

    def test_weighter_on_device(self) -> None:
        """LiLAWWeighter tensors can be moved to a device."""
        weighter = LiLAWWeighter()
        device = torch.device("cpu")
        weighter.alpha = weighter.alpha.to(device).requires_grad_(True)
        weighter.beta = weighter.beta.to(device).requires_grad_(True)
        weighter.delta = weighter.delta.to(device).requires_grad_(True)

        s_label = torch.rand(16)
        s_max = torch.rand(16)
        _, _, _, w = weighter.compute_weights(s_label, s_max)
        assert w.shape == (16,)


# ── Multi-class evaluation ───────────────────────────────────────────────────


class TestMultiClassEval:
    def test_accuracy_perfect(self) -> None:
        y_true = torch.tensor([0, 1, 2, 3])
        logits = torch.eye(4) * 10
        assert compute_accuracy(y_true, logits) == 1.0

    def test_accuracy_zero(self) -> None:
        y_true = torch.tensor([0, 1, 2, 3])
        # Shift logits so every prediction is wrong
        logits = torch.eye(4).roll(1, dims=1) * 10
        assert compute_accuracy(y_true, logits) == 0.0

    def test_auroc_perfect(self) -> None:
        y_true = torch.tensor([0, 1, 2])
        probs = torch.eye(3)
        auroc = compute_auroc(y_true, probs, 3)
        assert auroc == 1.0

    def test_auroc_handles_missing_class(self) -> None:
        """AUROC should return NaN or handle gracefully when a class is missing."""
        y_true = torch.tensor([0])  # Only one class present
        probs = torch.tensor([[1.0, 0.0]])
        auroc = compute_auroc(y_true, probs, 2)
        assert np.isnan(auroc)
