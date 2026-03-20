#!/usr/bin/env bash
# volume_deploy.sh — Deploy Dev Fleet from Modal Volume (no git clone needed).
#
# This script replaces the git-clone-based workflow:
#   OLD: git clone → modal deploy app.py
#   NEW: modal volume pull → modal deploy app.py
#
# Credentials
# ───────────
# Two sets of Modal credentials are expected as env vars:
#
#   MODAL_VOLUME_TOKEN_ID      — second account: manages dev-fleet-repo volume
#   MODAL_VOLUME_TOKEN_SECRET
#
#   MODAL_TOKEN_ID             — primary account: deploys the dev_fleet app
#   MODAL_TOKEN_SECRET
#
# Quick start (first time)
# ────────────────────────
#   1. Set all four env vars above.
#   2. Push current repo to the volume once:
#        ./volume_deploy.sh --sync-only
#   3. From now on, deploy entirely from the volume:
#        ./volume_deploy.sh
#
# Usage
# ─────
#   ./volume_deploy.sh             # pull from volume + deploy
#   ./volume_deploy.sh --sync      # push local → volume, then pull + deploy
#   ./volume_deploy.sh --sync-only # push local → volume (no deploy)
#   ./volume_deploy.sh --pull-only # pull volume → /tmp (no deploy)

set -euo pipefail

DEPLOY_DIR="${VOLUME_DEPLOY_DIR:-/tmp/dev-fleet-deploy}"
VOLUME_NAME="dev-fleet-repo"

SYNC=0
SYNC_ONLY=0
PULL_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --sync)       SYNC=1 ;;
    --sync-only)  SYNC=1; SYNC_ONLY=1 ;;
    --pull-only)  PULL_ONLY=1 ;;
    -h|--help)
      sed -n '/^# Usage/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ── validate credentials ──────────────────────────────────────────────────────

if [[ $SYNC -eq 1 ]]; then
  : "${MODAL_VOLUME_TOKEN_ID:?MODAL_VOLUME_TOKEN_ID is required for --sync}"
  : "${MODAL_VOLUME_TOKEN_SECRET:?MODAL_VOLUME_TOKEN_SECRET is required for --sync}"
fi

if [[ $SYNC_ONLY -eq 0 && $PULL_ONLY -eq 0 ]]; then
  # Pull always needs volume credentials
  : "${MODAL_VOLUME_TOKEN_ID:?MODAL_VOLUME_TOKEN_ID is required}"
  : "${MODAL_VOLUME_TOKEN_SECRET:?MODAL_VOLUME_TOKEN_SECRET is required}"
fi

if [[ $SYNC_ONLY -eq 0 && $PULL_ONLY -eq 0 ]]; then
  : "${MODAL_TOKEN_ID:?MODAL_TOKEN_ID is required for deployment}"
  : "${MODAL_TOKEN_SECRET:?MODAL_TOKEN_SECRET is required for deployment}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── step 1: push local repo → volume (optional) ───────────────────────────────

if [[ $SYNC -eq 1 ]]; then
  echo ">>> Syncing local repo → volume '${VOLUME_NAME}' …"
  MODAL_TOKEN_ID="$MODAL_VOLUME_TOKEN_ID" \
  MODAL_TOKEN_SECRET="$MODAL_VOLUME_TOKEN_SECRET" \
    python "$SCRIPT_DIR/volume_manager.py" push
  echo ">>> Sync complete."

  if [[ $SYNC_ONLY -eq 1 ]]; then
    exit 0
  fi
fi

# ── step 2: pull volume → local dir ───────────────────────────────────────────

if [[ $PULL_ONLY -eq 1 ]]; then
  echo ">>> Pulling volume '${VOLUME_NAME}' → ${DEPLOY_DIR} …"
  MODAL_TOKEN_ID="$MODAL_VOLUME_TOKEN_ID" \
  MODAL_TOKEN_SECRET="$MODAL_VOLUME_TOKEN_SECRET" \
    python "$SCRIPT_DIR/volume_manager.py" pull --dest "$DEPLOY_DIR"
  echo ">>> Pull complete: ${DEPLOY_DIR}"
  exit 0
fi

echo ">>> Pulling volume '${VOLUME_NAME}' → ${DEPLOY_DIR} …"
MODAL_TOKEN_ID="$MODAL_VOLUME_TOKEN_ID" \
MODAL_TOKEN_SECRET="$MODAL_VOLUME_TOKEN_SECRET" \
  python "$SCRIPT_DIR/volume_manager.py" pull --dest "$DEPLOY_DIR"
echo ">>> Pull complete."

# ── step 3: deploy from pulled directory ─────────────────────────────────────

echo ">>> Deploying Dev Fleet from volume snapshot …"
MODAL_TOKEN_ID="$MODAL_TOKEN_ID" \
MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" \
  modal deploy "$DEPLOY_DIR/app.py"

echo ">>> Dev Fleet deployed from volume.  Retrieve logs with:"
echo "  modal app logs dev_fleet"
