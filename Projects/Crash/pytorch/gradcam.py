"""Grad-CAM overlay generation.

Grad-CAM highlights the image regions that most drove the model's decision, the same
explainability idea as the LIME heatmaps in the old random-forest app but computed from
the network's own gradients. Uses the maintained pytorch-grad-cam package.

The classifier has a single-logit head, which pytorch-grad-cam does not infer targets
for automatically (it assumes a multi-class output). We wrap the model so its output is
2D and pass an explicit binary target for the predicted class.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


class _TwoDimWrapper(nn.Module):
    """Reshape the single logit to (N, 1) so pytorch-grad-cam is happy."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out


def _pick_target_layer(model):
    """Grad-CAM needs the last spatial conv layer. Pick the last 2D conv in the backbone."""
    target = None
    for module in model.backbone.modules():
        if isinstance(module, nn.Conv2d):
            target = module
    if target is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return target


def make_overlay(model, input_tensor: torch.Tensor, display_rgb: np.ndarray) -> Image.Image:
    """
    input_tensor: normalized (1,3,H,W) tensor the model consumes.
    display_rgb:  float32 (H,W,3) in [0,1], same H,W as input_tensor, for the overlay base.
    Returns a PIL image with the Grad-CAM heatmap superimposed on display_rgb.
    """
    target_layer = _pick_target_layer(model)

    # Explain the class the model actually predicted for this image.
    with torch.no_grad():
        pred_class = 1 if torch.sigmoid(model(input_tensor)).item() > 0.5 else 0

    wrapped = _TwoDimWrapper(model)
    targets = [BinaryClassifierOutputTarget(pred_class)]
    with GradCAM(model=wrapped, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(display_rgb, grayscale, use_rgb=True)
    return Image.fromarray(overlay)
