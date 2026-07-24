"""Central configuration for the car-damage classifier.

One dataclass holds every knob the training, evaluation, and serving code needs, so
runs are reproducible from a single object rather than scattered constants.
"""
from dataclasses import dataclass, field
from pathlib import Path

# The dataset ships with the repo: two folders of JPEGs, 1 = intact, 2 = crashed.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "Projects" / "Crash" / "Data" / "AccidentDetection"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# ImageNet statistics, used because the backbone was pretrained on ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Folder name -> (class index, human label). Class 1 is the positive (crashed) class.
CLASS_MAP = {"1": (0, "not_crashed"), "2": (1, "crashed")}


@dataclass
class Config:
    backbone: str = "resnet18"          # any timm model; resnet18 is small and CPU-fast
    img_size: int = 224
    batch_size: int = 32
    head_epochs: int = 12               # epochs training the head on frozen features
    finetune_epochs: int = 3            # extra epochs with the backbone unfrozen (GPU path)
    lr: float = 1e-3
    finetune_lr: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = 0.3
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    num_workers: int = 0                # 0 is safest on Windows
    finetune: bool = False              # unfreeze backbone; only worthwhile with a GPU
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    artifacts_dir: Path = field(default_factory=lambda: ARTIFACTS_DIR)

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "model.pt"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def curve_path(self) -> Path:
        return self.artifacts_dir / "training_curve.png"

    @property
    def report_path(self) -> Path:
        return self.artifacts_dir / "test_report.txt"
