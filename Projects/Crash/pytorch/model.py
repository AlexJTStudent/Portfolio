"""Model definition shared by training and serving.

A timm backbone produces a pooled feature vector; a small head turns that into a
single logit for binary classification. Keeping this in one module means the serving
code rebuilds the exact architecture the training code saved.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CarDamageModel(nn.Module):
    def __init__(self, backbone: str = "resnet18", dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        # num_classes=0 gives a feature extractor; global pooling is applied by timm.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)  # logits, shape (N,)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


def save_model(model: CarDamageModel, path, cfg, class_names: list[str], metrics: dict) -> None:
    """Persist everything serving needs to rebuild and run the model."""
    from config import IMAGENET_MEAN, IMAGENET_STD
    torch.save({
        "backbone": model.backbone_name,
        "dropout": cfg.dropout,
        "img_size": cfg.img_size,
        "state_dict": model.state_dict(),
        "class_names": class_names,          # index 0, index 1
        "norm_mean": IMAGENET_MEAN,
        "norm_std": IMAGENET_STD,
        "metrics": metrics,
    }, path)


def load_model(path, map_location="cpu"):
    """Rebuild a CarDamageModel from a checkpoint. Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = CarDamageModel(backbone=ckpt["backbone"], dropout=ckpt["dropout"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
