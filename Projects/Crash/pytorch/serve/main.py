"""FastAPI serving app for the PyTorch car-damage classifier.

Endpoints:
  GET  /health   liveness plus model and Claude-budget status
  POST /predict  base64 image in, prediction + Grad-CAM heatmap + optional Claude note

The JSON response matches the contract the portfolio homepage already consumes
(status, message, confidence, heatmap_image, claude_summary), so the frontend needs
only a URL change. FastAPI gives typed request/response models and automatic docs at
/docs, a step up from the previous Flask apps.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
from datetime import date
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

# The app is copied next to model.py/gradcam.py inside the image; support both layouts.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load_model          # noqa: E402
from gradcam import make_overlay      # noqa: E402

def _resolve_model_path() -> str:
    """Find the model: env override, then serve/model.pt (Docker), then ../artifacts (local)."""
    if os.environ.get("MODEL_PATH"):
        return os.environ["MODEL_PATH"]
    here = Path(__file__).resolve().parent
    for cand in (here / "model.pt", here.parent / "artifacts" / "model.pt"):
        if cand.exists():
            return str(cand)
    return str(here / "model.pt")  # will fail loudly at load, which is the right signal


MODEL_PATH = _resolve_model_path()

app = FastAPI(title="Car Damage Detector", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- Model load -----------------------------------------------------------------
model = None
class_names = ["not_crashed", "crashed"]
img_size = 224
norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)
try:
    print("Loading PyTorch model...", flush=True)
    model, ckpt = load_model(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]
    img_size = ckpt["img_size"]
    norm_mean = tuple(ckpt["norm_mean"])
    norm_std = tuple(ckpt["norm_std"])
    print(f"Model loaded: {ckpt['backbone']} ({class_names})", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"ERROR loading model: {e}", flush=True)
    model = None

_normalize = transforms.Normalize(norm_mean, norm_std)

# ---- Claude (optional, budgeted) -------------------------------------------------
try:
    import anthropic
    _key = os.environ.get("ANTHROPIC_API_KEY")
    claude = anthropic.Anthropic(api_key=_key) if _key else None
    if claude:
        print("Anthropic client initialized", flush=True)
except Exception:  # noqa: BLE001
    claude = None

CLAUDE_DAILY_LIMIT = int(os.environ.get("CLAUDE_DAILY_LIMIT", "100"))
_claude_lock = threading.Lock()
_claude_calls = {"day": date.today(), "count": 0}


def _claude_budget_available() -> bool:
    with _claude_lock:
        today = date.today()
        if _claude_calls["day"] != today:
            _claude_calls["day"] = today
            _claude_calls["count"] = 0
        if _claude_calls["count"] >= CLAUDE_DAILY_LIMIT:
            return False
        _claude_calls["count"] += 1
        return True


def analyze_with_claude(heatmap_b64: str, label: str, confidence: float) -> str:
    if claude is None:
        return "Claude API not configured"
    if not _claude_budget_available():
        return ("The written analysis is limited to a fixed number of requests per day and "
                "today's allowance has been used. The prediction, confidence, and Grad-CAM "
                "heatmap above are unaffected.")
    try:
        data = heatmap_b64.split(",", 1)[1] if "," in heatmap_b64 else heatmap_b64
        prompt = (
            f"You are reviewing a car-damage model result. The model predicted "
            f"{label.upper()} at {confidence:.1f}% confidence. The image is a Grad-CAM "
            f"overlay where warm (red/yellow) regions are what the network weighed most. "
            f"In 3-4 sentences: what is in the highlighted regions, do you agree with the "
            f"prediction, and could the focus be misleading?"
        )
        msg = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return msg.content[0].text
    except Exception as e:  # noqa: BLE001
        print(f"Claude error: {e}", flush=True)
        return ("The written analysis is unavailable right now. The prediction, confidence, "
                "and Grad-CAM heatmap above are unaffected.")


# ---- Schemas ---------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # data URL or bare base64


class PredictResponse(BaseModel):
    status: str
    message: str
    confidence: float
    model_type: str
    heatmap_image: str | None = None
    claude_summary: str | None = None


# ---- Helpers ---------------------------------------------------------------------
def _decode_image(image_field: str) -> Image.Image:
    raw = image_field.split(",", 1)[1] if "," in image_field else image_field
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def _to_jpeg_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- Routes ----------------------------------------------------------------------
@app.get("/")
@app.get("/health")
def health():
    with _claude_lock:
        used = _claude_calls["count"] if _claude_calls["day"] == date.today() else 0
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": f"PyTorch transfer learning ({ckpt['backbone'] if model else 'n/a'})",
        "claude_api_configured": claude is not None,
        "claude_calls_used_today": used,
        "claude_daily_limit": CLAUDE_DAILY_LIMIT,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        return PredictResponse(status="error", message="Model not loaded",
                               confidence=0.0, model_type="none")
    # Decode and preprocess.
    pil = _decode_image(req.image)
    resized = pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized).astype("float32") / 255.0  # (H,W,3) in [0,1] for overlay
    tensor = _normalize(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        prob_crashed = torch.sigmoid(model(tensor)).item()
    pred = 1 if prob_crashed > 0.5 else 0
    confidence = (prob_crashed if pred == 1 else 1 - prob_crashed) * 100
    label = class_names[pred]

    # Grad-CAM needs gradients, so it runs outside the no_grad block.
    try:
        overlay = make_overlay(model, tensor, arr)
        heatmap_url = _to_jpeg_data_url(overlay)
    except Exception as e:  # noqa: BLE001
        print(f"Grad-CAM error: {e}", flush=True)
        heatmap_url = None

    summary = analyze_with_claude(heatmap_url, label, confidence) if heatmap_url else \
        "Heatmap unavailable, so no written analysis was produced."

    crashed = pred == 1
    return PredictResponse(
        status="crashed" if crashed else "not_crashed",
        message=("This car appears to be CRASHED or DAMAGED." if crashed
                 else "This car appears to be in good condition."),
        confidence=round(confidence, 2),
        model_type=f"PyTorch transfer learning ({ckpt['backbone']})",
        heatmap_image=heatmap_url,
        claude_summary=summary,
    )
