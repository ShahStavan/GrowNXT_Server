#!/bin/bash
set -e

echo "========================================================"
echo " Starting GrowNXT Server Platform"
echo "========================================================"

# Determine execution mode: Gradio UI + Flask, or Caddy + Gunicorn
MODE="${APP_MODE:-gradio}"

if [ "$MODE" = "caddy" ]; then
    echo "[GrowNXT] Launching Caddy Reverse Proxy on port ${PORT:-7860}..."
    caddy start --config /app/Caddyfile

    echo "[GrowNXT] Launching Flask Gunicorn Server on internal port 5000..."
    exec gunicorn --bind 127.0.0.1:5000 --workers 2 --timeout 300 api.app:app
else
    echo "[GrowNXT] Launching Gradio Interactive Platform on port ${PORT:-7860}..."
    exec python app.py
fi
