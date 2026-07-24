"""Export the trained model to ONNX.

The old repo started an ONNX conversion path and never finished it. This completes it:
a portable, framework-independent artifact that ONNX Runtime (or many other runtimes)
can serve without PyTorch installed. The live server still runs the eager PyTorch model
because Grad-CAM needs autograd, but the ONNX file is a useful portability artifact.

Usage:
    python export_onnx.py
"""
from __future__ import annotations

import torch

from config import Config
from model import load_model


def main():
    cfg = Config()
    if not cfg.model_path.exists():
        raise SystemExit(f"No model at {cfg.model_path}. Run train.py first.")

    model, ckpt = load_model(cfg.model_path, map_location="cpu")
    model.eval()
    out_path = cfg.artifacts_dir / "model.onnx"
    dummy = torch.randn(1, 3, ckpt["img_size"], ckpt["img_size"])

    # dynamo=False uses the legacy TorchScript exporter, which needs no extra packages
    # (torch 2.9's default dynamo path pulls in onnxscript).
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["image"], output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Exported {out_path.name} ({size_mb:.1f} MB)")

    # Verify the export loads and runs if onnxruntime is available.
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path))
        out = sess.run(None, {"image": dummy.numpy().astype(np.float32)})
        print(f"onnxruntime check OK, logit shape {np.shape(out[0])}")
    except ImportError:
        print("onnxruntime not installed; skipped the runtime check.")


if __name__ == "__main__":
    main()
