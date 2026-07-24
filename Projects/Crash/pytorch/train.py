"""Train the car-damage classifier with transfer learning.

Default path (CPU-friendly): freeze the pretrained backbone, extract features once,
and train only the classification head. This is real transfer learning and finishes
in minutes on a CPU. Pass --finetune to additionally unfreeze the backbone for a few
low-learning-rate epochs, which is worthwhile only on a GPU.

Usage:
    python train.py                       # frozen backbone, resnet18
    python train.py --backbone efficientnet_b0
    python train.py --finetune            # GPU recommended
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CLASS_MAP, Config
from data import make_datasets
from model import CarDamageModel, save_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_features(model: CarDamageModel, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen backbone once and cache pooled feature vectors."""
    model.backbone.eval()
    feats, labels = [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        feats.append(model.backbone(x).cpu())
        labels.append(y)
        print(f"    features {min((i + 1) * loader.batch_size, len(loader.dataset))}"
              f"/{len(loader.dataset)}", end="\r", flush=True)
    print()
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def evaluate_logits(model: CarDamageModel, loader: DataLoader, device) -> tuple[float, float]:
    """Return (accuracy, mean BCE loss) over a loader."""
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        n_batches += 1
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.numel()
    return correct / total, loss_sum / max(n_batches, 1)


def train_head_on_features(model, feats, labels, val_loader, cfg, device):
    """Phase 1: train just the head on cached features. Fast, CPU-friendly."""
    head = model.head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.BCEWithLogitsLoss()
    ds = TensorDataset(feats, labels.float())
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    history = {"val_acc": [], "val_loss": [], "train_loss": []}
    best_acc, best_state = 0.0, None
    for epoch in range(cfg.head_epochs):
        head.train()
        running = 0.0
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            opt.zero_grad()
            logits = head(fb).squeeze(1)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()
        val_acc, val_loss = evaluate_logits(model, val_loader, device)
        history["train_loss"].append(running / len(loader))
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        print(f"  [head {epoch + 1:2d}/{cfg.head_epochs}] "
              f"train_loss={running / len(loader):.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
    if best_state is not None:
        model.head.load_state_dict(best_state)
    return history, best_acc


def finetune_full(model, train_loader, val_loader, cfg, device, history):
    """Phase 2 (optional): unfreeze the backbone and fine-tune end to end."""
    model.unfreeze_backbone()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    crit = nn.BCEWithLogitsLoss()
    best_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(cfg.finetune_epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
        val_acc, val_loss = evaluate_logits(model, val_loader, device)
        history["train_loss"].append(running / len(train_loader))
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        print(f"  [fine {epoch + 1:2d}/{cfg.finetune_epochs}] "
              f"train_loss={running / len(train_loader):.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history, best_acc


def plot_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["val_acc"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, history["val_acc"], marker="o")
    ax[0].set_title("Validation accuracy"); ax[0].set_xlabel("epoch"); ax[0].grid(True)
    ax[1].plot(epochs, history["train_loss"], label="train")
    ax[1].plot(epochs, history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=None)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--head-epochs", type=int, default=None)
    args = p.parse_args()

    cfg = Config()
    if args.backbone:
        cfg.backbone = args.backbone
    if args.finetune:
        cfg.finetune = True
    if args.head_epochs:
        cfg.head_epochs = args.head_epochs

    torch.manual_seed(cfg.seed)
    device = get_device()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | backbone: {cfg.backbone} | finetune: {cfg.finetune}")

    ds = make_datasets(cfg)
    print(f"Split: train={len(ds['train'])}  val={len(ds['val'])}  test={len(ds['test'])}")

    model = CarDamageModel(backbone=cfg.backbone, dropout=cfg.dropout, pretrained=True).to(device)
    model.freeze_backbone()

    val_loader = DataLoader(ds["val"], batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    t0 = time.time()
    print("Phase 1: caching frozen-backbone features...")
    feat_loader = DataLoader(ds["train_eval"], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    feats, labels = extract_features(model, feat_loader, device)
    print(f"Cached features: {tuple(feats.shape)}")

    print("Phase 1: training the head...")
    history, best_acc = train_head_on_features(model, feats, labels, val_loader, cfg, device)

    if cfg.finetune:
        print("Phase 2: fine-tuning the full network...")
        train_loader = DataLoader(ds["train"], batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers)
        history, best_acc = finetune_full(model, train_loader, val_loader, cfg, device, history)

    elapsed = time.time() - t0
    class_names = [CLASS_MAP["1"][1], CLASS_MAP["2"][1]]  # index 0, index 1
    metrics = {
        "backbone": cfg.backbone,
        "best_val_acc": round(best_acc, 4),
        "epochs_run": len(history["val_acc"]),
        "finetuned": cfg.finetune,
        "train_seconds": round(elapsed, 1),
    }
    save_model(model, cfg.model_path, cfg, class_names, metrics)
    cfg.metrics_path.write_text(json.dumps(metrics, indent=2))
    plot_curve(history, cfg.curve_path)

    print(f"\nDone in {elapsed:.1f}s. Best val acc: {best_acc:.4f}")
    print(f"Saved: {cfg.model_path.name}, {cfg.metrics_path.name}, {cfg.curve_path.name}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
