# MedMNIST Replication Experiment

Replicates the LiLAW paper's MedMNIST experiment to validate our implementation against their published results.

## Motivation

Our tabular PoC (breast_cancer, adult, pima) showed only 1-2% PR-AUC gains, far below the paper's reported 17-40% improvements on image datasets. This replication uses the paper's exact experimental setup to determine whether the gap is due to (a) our implementation, or (b) the difference between tabular and image data.

## Paper Settings (MedMNIST)

| Parameter | Value |
|---|---|
| Model | ResNet-18, ImageNet-21K pretrained (via `timm`) |
| Optimizer | Adam (lr=0.0001, weight_decay=0) |
| LR scheduler | MultiStepLR (×0.1 at epochs 50, 75) |
| Epochs | 100 |
| Batch size | 128 |
| Input size | 224×224 |
| Noise type | Symmetric (uniform random label replacement) |
| Warmup | 1 epoch vanilla CE |
| LiLAW params | α=10, β=2, δ=6, meta_lr=0.005, meta_wd=0.0001 |
| Early stopping | Patience 10, on clean val accuracy |
| Metrics | Top-1 accuracy, macro AUROC |

## Our Sweep Configuration

| Parameter | Value |
|---|---|
| Datasets | BloodMNIST (11,959 train, 8 classes) |
| Noise rates | 0%, 20%, 40%, 60%, 80% |
| Seeds | 42, 123, 456 |
| Total runs | 30 (5 noise rates × 3 seeds × 2 methods) |

## How to Run

### Local (CPU, for testing)
```bash
cd lilaw-poc
pip install -e ".[medmnist]"
python -c "
from lilaw_poc.medmnist.experiment import run_sweep
run_sweep(epochs=2)  # smoke test
"
```

### RunPod (GPU)
```bash
# Ensure runpodctl is configured and GH_TOKEN is set
export GH_TOKEN=$(gh auth token)
bash scripts/run_medmnist_sweep.sh [BRANCH]
```

The script creates an H100 SXM pod, runs the full sweep (~40-55 min), downloads results, and deletes the pod. Estimated cost: ~$2-3.

## Expected Results (from paper)

The paper reports Top-1 accuracy for BloodMNIST under symmetric noise. At 0% noise, both baseline and LiLAW should achieve ~95%+ accuracy. At higher noise rates (40-80%), the paper shows LiLAW maintaining significantly higher accuracy than baseline CE, with gaps of 5-15+ percentage points.

If our results match these trends, the implementation is validated. If not, the implementation has a bug that was masked by the modest tabular-data signal.

## Deviations from Paper

1. **3 seeds** instead of 5 (compute savings; sufficient for trend validation)
2. **5 noise rates** instead of 10 (every other step)
3. **1 dataset** instead of 10 (BloodMNIST as representative; expandable to PathMNIST)
4. **ImageNet-1K weights from timm** — `timm`'s default `resnet18` pretrained weights are ImageNet-1K. The paper specifies ImageNet-21K. If results diverge, switch to a 21K checkpoint.
