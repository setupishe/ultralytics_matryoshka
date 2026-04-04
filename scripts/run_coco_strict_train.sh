#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_YOLO="/home/setupishe/bel_conf/.venv/bin/yolo"

if [[ ! -x "${VENV_YOLO}" ]]; then
  echo "Missing yolo CLI at ${VENV_YOLO}" >&2
  exit 1
fi

VARIANT="${1:-vanilla}"
shift || true

FRACTION="${FRACTION:-0.2}"
PROJECT="${PROJECT:-${ROOT_DIR}/runs/detect/strict_parity}"
EPOCHS="${EPOCHS:-65}"
BASE_NAME="COCO_random_${FRACTION}_m"

declare -a EXTRA_ARGS=()

case "${VARIANT}" in
  vanilla)
    RUN_NAME="${RUN_NAME:-${BASE_NAME}}"
    ;;
  fastsafe)
    RUN_NAME="${RUN_NAME:-${BASE_NAME}_fastsafe}"
    EXTRA_ARGS+=("plots=False" "verbose=False")
    ;;
  workers10)
    RUN_NAME="${RUN_NAME:-${BASE_NAME}_workers10}"
    EXTRA_ARGS+=("workers=10")
    ;;
  workers12)
    RUN_NAME="${RUN_NAME:-${BASE_NAME}_workers12}"
    EXTRA_ARGS+=("workers=12")
    ;;
  cache-ram)
    RUN_NAME="${RUN_NAME:-${BASE_NAME}_cache_ram}"
    EXTRA_ARGS+=("cache=ram")
    ;;
  cache-disk)
    RUN_NAME="${RUN_NAME:-${BASE_NAME}_cache_disk}"
    EXTRA_ARGS+=("cache=disk")
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    echo "Supported variants: vanilla, fastsafe, workers10, workers12, cache-ram, cache-disk" >&2
    exit 1
    ;;
esac

DATA_CFG="ultralytics/cfg/datasets/COCO_${FRACTION}.yaml"
if [[ ! -f "${ROOT_DIR}/${DATA_CFG}" ]]; then
  echo "Missing dataset config: ${ROOT_DIR}/${DATA_CFG}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

CMD=(
  "${VENV_YOLO}"
  detect
  train
  "data=${DATA_CFG}"
  "model=yolov8m.pt"
  "imgsz=640"
  "batch=48"
  "epochs=${EPOCHS}"
  "pretrained=False"
  "name=${RUN_NAME}"
  "project=${PROJECT}"
)
CMD+=("${EXTRA_ARGS[@]}")
CMD+=("$@")

printf 'Running command:\n  '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
