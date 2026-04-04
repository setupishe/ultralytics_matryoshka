#!/usr/bin/env bash
set -euo pipefail

# Log-only training monitor for Ultralytics runs.
# Usage: ./monitor_train.sh [interval_seconds] [log_file]

INTERVAL="${1:-1}"
LOG_FILE="${2:-monitor_$(date +%Y%m%d_%H%M%S).log}"

echo "Starting monitor logger (interval=${INTERVAL}s)..."
echo "Keep training in another terminal."
echo "Writing snapshots to: ${LOG_FILE}"
echo "Press Ctrl+C to stop."
echo ""

while true; do
  {
    echo "=== $(date --iso-8601=seconds) ==="
    echo "--- nvidia-smi ---"
    nvidia-smi || true
    echo
    echo "--- CPU (top 15 processes by CPU) ---"
    ps -eo pid,ppid,ni,%cpu,%mem,etime,cmd --sort=-%cpu | sed -n '1,16p'
    echo
    echo "--- Disk I/O ---"
    if command -v iostat >/dev/null 2>&1; then
      iostat -xz 1 1 || true
    else
      echo "iostat not found (install sysstat to enable disk metrics)."
    fi
    echo
  } >>"${LOG_FILE}"
  echo "[$(date +%H:%M:%S)] snapshot appended to ${LOG_FILE}"
  sleep "${INTERVAL}"
done
