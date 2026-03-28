import torch

from lilaw_poc.model import MLMDNet


class TestMLMDNet:
    """Test the 5-layer MLP matching MLMD architecture."""

    def test_output_shape(self) -> None:
        """Output should be (batch_size, 1)."""
        model = MLMDNet(input_dim=30)
        x = torch.randn(16, 30)
        out = model(x)
        assert out.shape == (16, 1)

    def test_output_range_sigmoid(self) -> None:
        """Output should be in (0, 1) due to sigmoid."""
        model = MLMDNet(input_dim=30)
        x = torch.randn(100, 30)
        out = model(x)
        assert (out > 0).all() and (out < 1).all()

    def test_has_five_linear_layers(self) -> None:
        """Architecture must have exactly 5 Linear layers."""
        model = MLMDNet(input_dim=30)
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 5

    def test_gradient_flows(self) -> None:
        """Loss backprop should produce gradients on all parameters."""
        model = MLMDNet(input_dim=10)
        x = torch.randn(4, 10)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert not torch.all(p.grad == 0)
