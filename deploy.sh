#!/usr/bin/env bash
# deploy.sh — Deploy Dev Fleet on Modal.
#
# Usage:
#   ./deploy.sh
#
# Logs:
#   modal app logs dev_fleet

set -euo pipefail

echo ">>> Deploying Dev Fleet (inference + reranker + orchestrator) …"
modal deploy app.py
echo ">>> Dev Fleet deployed.  Retrieve logs with:"
echo "  modal app logs dev_fleet"
