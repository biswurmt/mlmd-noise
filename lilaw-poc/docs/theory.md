# LiLAW Theory

The LiLAW method assigns a weight to each sample based on prediction confidence.
For sample $i$, let $s_{i,y_i}$ denote the model's predicted probability of the correct label.

## Weight Components

- **Wα (easy samples)**: $\sigma(\alpha \cdot s_{y_i} - \max(s))$  
  Down-weights high-confidence correct predictions.

- **Wβ (hard samples)**: $\sigma(-(\beta \cdot s_{y_i} - \max(s))$)  
  Down-weights low-confidence predictions (likely noisy).

- **Wδ (moderate)**: RBF kernel centered near the decision boundary:  
  $\exp(-\frac12(\delta \cdot s_{y_i} - \max(s))^2)$

Total weight: $W_i = W_\alpha + W_\beta + W_\delta$, applied multiplicatively to per-sample BCE loss.

The parameters α, β, δ are updated by gradient descent on a validation mini-batch, using the weighted loss as meta-objective.

For full math, see the [LiLAW paper](https://arxiv.org/abs/2502.01981) and [DECISIONS.md](DECISIONS.md).
