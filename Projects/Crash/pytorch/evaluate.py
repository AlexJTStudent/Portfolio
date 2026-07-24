"""Evaluate the trained model on the held-out test split.

Reports accuracy, precision, recall, F1, and ROC-AUC plus a confusion matrix, and
writes them to artifacts/test_report.txt. These are the honest numbers the portfolio
writeup cites against the old random forest.
"""
from __future__ import annotations

import json

import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from config import Config
from data import make_datasets
from model import load_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu()
        probs.append(p)
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.model_path.exists():
        raise SystemExit(f"No model at {cfg.model_path}. Run train.py first.")

    model, ckpt = load_model(cfg.model_path, map_location=device)
    model.to(device)
    class_names = ckpt["class_names"]

    ds = make_datasets(cfg)
    test_loader = DataLoader(ds["test"], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    probs, y_true = collect_predictions(model, test_loader, device)
    y_pred = (probs > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, y_pred)

    lines = [
        "Car Damage Classifier - Test Set Evaluation",
        "=" * 44,
        f"Backbone     : {ckpt['backbone']}",
        f"Test images  : {len(y_true)}",
        f"Classes      : 0={class_names[0]}, 1={class_names[1]}",
        "",
        f"Accuracy     : {acc:.4f}",
        f"Precision    : {prec:.4f}   (of predicted-crashed, how many were crashed)",
        f"Recall       : {rec:.4f}   (of actual-crashed, how many we caught)",
        f"F1           : {f1:.4f}",
        f"ROC-AUC      : {auc:.4f}",
        "",
        "Confusion matrix (rows = actual, cols = predicted):",
        f"                pred_{class_names[0]:<12} pred_{class_names[1]}",
        f"  actual_{class_names[0]:<10} {cm[0, 0]:<17} {cm[0, 1]}",
        f"  actual_{class_names[1]:<10} {cm[1, 0]:<17} {cm[1, 1]}",
    ]
    report = "\n".join(lines)
    print(report)
    cfg.report_path.write_text(report)

    summary = {"accuracy": round(acc, 4), "precision": round(prec, 4),
               "recall": round(rec, 4), "f1": round(f1, 4), "roc_auc": round(auc, 4),
               "test_n": int(len(y_true))}
    # Merge into metrics.json so the writeup can read one file.
    metrics = json.loads(cfg.metrics_path.read_text()) if cfg.metrics_path.exists() else {}
    metrics["test"] = summary
    cfg.metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {cfg.report_path.name} and updated {cfg.metrics_path.name}")


if __name__ == "__main__":
    main()
