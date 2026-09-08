"""
data.py  —  CIFAR-10 loading and preprocessing.

Train augmentation pipeline (Q1a):
  1. RandomCrop(32, padding=4)    — translation invariance via padding + random crop
  2. RandomHorizontalFlip()       — valid for all CIFAR-10 classes (no left/right semantics)
  3. ColorJitter(...)             — colour/brightness robustness; applied stochastically
  4. ToTensor()                   — PIL [0,255] uint8  →  float32 [0,1], CHW layout
  5. Normalize(mean, std)         — per-channel standardisation using CIFAR-10 train stats
  6. RandomErasing(p=0.1)        — occlusion robustness; applied in tensor space

Test pipeline: ToTensor + Normalize only (no augmentation; we want clean accuracy).
"""

import torch
from torchvision import datasets, transforms

# Per-channel mean and std of the CIFAR-10 training set (standard constants).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms():
    """Returns (train_transform, test_transform)."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.20)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform


def get_dataloaders(data_dir="./cifar10_data", batch_size=128, num_workers=4):
    """Downloads CIFAR-10 if needed and returns (train_loader, test_loader)."""
    train_tf, test_tf = get_transforms()
    train_set = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    _kw = dict(pin_memory=True, persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, **_kw
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, **_kw
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")   # [128, 3, 32, 32]
    print(f"Label shape : {labels.shape}")   # [128]
    print(f"Pixel range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"Train size  : {len(train_loader.dataset)}")
    print(f"Test size   : {len(test_loader.dataset)}")
