"""MedMNIST dataset loading with noise injection support."""

from typing import Literal

import medmnist
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MedMNISTName = Literal[
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NoisyLabelDataset(Dataset):
    """Wraps a dataset and replaces labels with noisy versions."""

    def __init__(self, base_dataset: Dataset, noisy_labels: torch.Tensor) -> None:
        self.base = base_dataset
        self.noisy_labels = noisy_labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, _ = self.base[idx]
        return img, int(self.noisy_labels[idx].item())

    def __len__(self) -> int:
        return len(self.base)


def _ensure_3ch(img: torch.Tensor) -> torch.Tensor:
    """Repeat single-channel image to 3 channels."""
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img


def get_medmnist_loaders(
    dataset_name: MedMNISTName,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load a MedMNIST dataset with standard transforms.

    Uses MedMNIST's official train/val/test splits.

    Args:
        dataset_name: Name of the MedMNIST dataset.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        image_size: Target image size (default 224 for ResNet).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes).
    """
    info = medmnist.INFO[dataset_name]
    num_classes = len(info["label"])
    dataset_cls = getattr(medmnist, info["python_class"])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_ensure_3ch),
        transforms.Resize(image_size, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_ensure_3ch),
        transforms.Resize(image_size, antialias=True),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = dataset_cls(split="train", transform=train_transform, download=True)
    val_ds = dataset_cls(split="val", transform=eval_transform, download=True)
    test_ds = dataset_cls(split="test", transform=eval_transform, download=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader, num_classes


def get_labels(dataset: Dataset) -> torch.Tensor:
    """Extract all labels from a MedMNIST dataset as a 1-D long tensor."""
    labels = dataset.labels  # type: ignore[attr-defined]
    return torch.from_numpy(labels).squeeze().long()
