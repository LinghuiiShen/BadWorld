#!/usr/bin/env bash
# Simple entrypoint: activate venv if present and run provided command
set -euo pipefail

if [ -f "/workspace/venv/bin/activate" ]; then
  echo "Activating venv"
  # shellcheck disable=SC1091
  source /workspace/venv/bin/activate
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec bash
fi
