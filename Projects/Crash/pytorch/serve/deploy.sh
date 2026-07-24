#!/bin/bash
set -euo pipefail

# Car Damage Detector (PyTorch) - Cloud Run deployment.
#
# Deploys as a NEW service, car-crash-detector-torch, so the existing random-forest
# service keeps serving until the frontend is switched over.
#
# Cost posture matches the RF service: scales to zero when idle (free), hard instance
# cap so a traffic spike cannot fan out, timeout aligned with gunicorn.
#
# The Anthropic key comes from Secret Manager and is never stored here. Create it with:
#   gcloud secrets create anthropic-api-key --replication-policy=automatic
#   printf '%s' 'YOUR_KEY' | gcloud secrets versions add anthropic-api-key --data-file=-

SERVICE="car-crash-detector-torch"
REGION="us-central1"
SECRET="anthropic-api-key"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "🚗 Car Damage Detector (PyTorch) Deployment"
echo "==========================================="

# The model and the two shared modules live one level up; stage copies next to the
# Dockerfile so the build context is self-contained, then clean them up on exit.
STAGED=()
cleanup() { for f in "${STAGED[@]:-}"; do [ -n "$f" ] && rm -f "$HERE/$f"; done; }
trap cleanup EXIT

for src in "../model.py" "../gradcam.py" "../artifacts/model.pt"; do
  base="$(basename "$src")"
  if [ ! -f "$HERE/$src" ]; then
    echo "❌ ERROR: missing $src (did training run? expected artifacts/model.pt)"
    exit 1
  fi
  cp "$HERE/$src" "$HERE/$base"
  STAGED+=("$base")
done
echo "✓ Staged model.py, gradcam.py, model.pt"

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "❌ ERROR: no gcloud project set. Run: gcloud config set project crashdetector-478006"
  exit 1
fi
echo "✓ Project: $PROJECT"

SECRET_FLAG=()
if gcloud secrets describe "$SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  SECRET_FLAG=(--set-secrets "ANTHROPIC_API_KEY=${SECRET}:latest")
  echo "✓ Secret '$SECRET' found, Claude analysis enabled"
else
  echo "⚠ Secret '$SECRET' not found. Deploying without it; predictions and Grad-CAM"
  echo "  still work, Claude commentary reads 'Claude API not configured'."
fi

echo ""
echo "📦 Deploying to Google Cloud Run..."
echo ""

gcloud run deploy "$SERVICE" \
  --source "$HERE" \
  --quiet \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  ${SECRET_FLAG[@]+"${SECRET_FLAG[@]}"}

echo ""
echo "✅ Deployment complete"
URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "Service URL: $URL"
echo ""
echo "Health check (first call cold-starts, allow up to 60s):"
curl -s --max-time 120 "$URL/health" || echo "  no response yet, retry in a moment"
echo ""
echo ""
echo "Point the homepage at: $URL"
