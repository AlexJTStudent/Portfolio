# Car Damage Detector, PyTorch rewrite

A rebuild of the wrecked-car classifier using transfer learning in PyTorch and a
FastAPI serving layer, replacing an earlier hand-crafted random forest served from
Flask. The point of the rewrite was to move the project onto the tools that image
classification is actually built with today, and to measure the result on a proper
held-out test set rather than reporting a number from training data.

## Result

On a held-out test split of 734 images the model never saw during training:

| Metric | Value |
|---|---|
| Accuracy | 94.6% |
| Precision | 95.7% |
| Recall | 93.0% |
| F1 | 0.94 |
| ROC-AUC | 0.991 |

Confusion matrix (rows are the truth, columns are the prediction):

|  | predicted intact | predicted crashed |
|---|---|---|
| actually intact | 360 | 15 |
| actually crashed | 25 | 334 |

The two error types are close to balanced, which is what you want when neither a false
alarm nor a missed crash is obviously cheaper than the other.

## What changed from the old version

The previous production model turned each image into 48 hand-designed numbers, colour
histograms, edge gradients, and patch-variance texture, and fed them to a random forest.
That throws away almost everything about where things are in the image. Its reported
accuracy came from training data, with no clean held-out test.

This version fine-tunes a network that was pretrained on ImageNet, so it starts already
knowing generic visual features and only has to learn the crashed-versus-intact
distinction. The comparison is not perfectly apples to apples, since the old number was
never measured on a held-out split, but the honest read is that a pretrained backbone on
a real test set lands at 94.6% where the hand-crafted approach had no trustworthy figure
at all.

## How it works

- **Backbone**: a pretrained model from `timm` (default `resnet18`), used as a frozen
  feature extractor. `efficientnet_b0` and `resnet50` are drop-in alternatives.
- **Head**: a small two-layer classifier trained on top of the frozen features.
- **Training** freezes the backbone, caches its features in a single pass, then trains
  only the head. That is what makes this practical on a CPU: the full run above took
  about 6.7 minutes with no GPU. Passing `--finetune` unfreezes the backbone for a few
  low-learning-rate epochs, which is worth doing on a GPU (see the Colab note below).
- **Explainability**: Grad-CAM overlays the regions the network weighed most, the same
  idea as the LIME heatmaps in the old app but read straight from the model's gradients.
- **Optional written analysis**: the Grad-CAM image is sent to Claude for a short
  plain-language read of what the model focused on. This is capped to a fixed number of
  calls per day and degrades to a static message if the budget is hit or the key is
  missing, so it can never run up a surprise bill or break a prediction.

## Layout

```
config.py        run configuration in one dataclass
data.py          dataset loading and a stratified train/val/test split
model.py         the timm backbone plus classifier head, shared by training and serving
train.py         transfer-learning training, saves artifacts/model.pt
evaluate.py      held-out test metrics and confusion matrix
gradcam.py       Grad-CAM overlay generation
export_onnx.py   export the trained model to ONNX (portable artifact)
serve/           FastAPI app, Dockerfile, and Cloud Run deploy script
artifacts/       trained model, metrics, training curve, test report
```

## Running it

Install the training dependencies (torch comes from the CPU wheel index):

```bash
pip install -r requirements-train.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

Train, evaluate, and serve:

```bash
python train.py           # writes artifacts/model.pt in a few minutes on CPU
python evaluate.py        # writes artifacts/test_report.txt
uvicorn serve.main:app    # interactive API docs at http://localhost:8000/docs
```

The training data (the published Pashaei, Ghatee, and Sajedi accident dataset) ships in
the repo under `Projects/Crash/Data/AccidentDetection`, so a clean clone reproduces the
run with no external download.

## Serving and deployment

The API is FastAPI, which gives typed request and response models and interactive docs at
`/docs` for free. It exposes `/health` and `/predict`, returning the same JSON shape the
portfolio homepage already consumes, so switching the site to this backend is a one-line
URL change.

Deployment targets Google Cloud Run through `serve/deploy.sh`, with the same cost controls
as the rest of the project: it scales to zero when idle, so it costs nothing to keep
online, and it is capped at three instances so a traffic spike cannot fan out. The
container runs a single worker to keep memory bounded.

## GPU fine-tuning

The frozen-backbone result above is already strong. To squeeze out more, run with
`--finetune` on a GPU, for example a free Colab runtime: upload this folder and the
dataset, `pip install -r requirements-train.txt`, then `python train.py --finetune`. The
saved artifact format is identical, so the serving code needs no change.
