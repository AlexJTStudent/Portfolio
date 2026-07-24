"""Dataset loading and stratified splitting for the AccidentDetection images.

The two source folders (1 = intact, 2 = crashed) are read once, then split into
train / val / test with a fixed seed so every run sees the same partition. Training
images get light augmentation; val and test get a deterministic resize and crop.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_MAP, IMAGENET_MEAN, IMAGENET_STD, Config

_EXTS = (".jpg", ".jpeg", ".png")


def _list_samples(data_dir: Path) -> list[tuple[str, int]]:
    """Return (path, label) pairs, reading the folder->class mapping from config."""
    samples: list[tuple[str, int]] = []
    for folder, (label, _name) in CLASS_MAP.items():
        d = data_dir / folder
        if not d.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {d}")
        for name in os.listdir(d):
            if name.lower().endswith(_EXTS):
                samples.append((str(d / name), label))
    if not samples:
        raise RuntimeError(f"No images found under {data_dir}")
    return samples


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageList(Dataset):
    """A plain (path, label) dataset. Corrupt images fall back to a black frame."""

    def __init__(self, samples: list[tuple[str, int]], tfm: transforms.Compose, img_size: int):
        self.samples = samples
        self.tfm = tfm
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size))
        return self.tfm(img), label


def stratified_split(cfg: Config):
    """Split samples into train/val/test, preserving class balance."""
    samples = _list_samples(cfg.data_dir)
    g = torch.Generator().manual_seed(cfg.seed)

    by_class: dict[int, list[tuple[str, int]]] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train, val, test = [], [], []
    for label, items in by_class.items():
        perm = torch.randperm(len(items), generator=g).tolist()
        items = [items[i] for i in perm]
        n = len(items)
        n_test = int(n * cfg.test_frac)
        n_val = int(n * cfg.val_frac)
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    return train, val, test


def make_datasets(cfg: Config):
    train, val, test = stratified_split(cfg)
    train_ds = ImageList(train, build_transforms(cfg.img_size, train=True), cfg.img_size)
    # Eval transform (no augmentation) for the training set too, used by feature caching.
    train_eval_ds = ImageList(train, build_transforms(cfg.img_size, train=False), cfg.img_size)
    val_ds = ImageList(val, build_transforms(cfg.img_size, train=False), cfg.img_size)
    test_ds = ImageList(test, build_transforms(cfg.img_size, train=False), cfg.img_size)
    return {"train": train_ds, "train_eval": train_eval_ds, "val": val_ds, "test": test_ds}
