#!/usr/bin/env bash
# deploy.sh — Deploy both Dev Fleet microservices on Modal (detached).
#
# Usage:
#   ./deploy.sh              # deploy both services
#   ./deploy.sh inference    # deploy only the inference engine
#   ./deploy.sh orchestrator # deploy only the orchestrator
#
# Logs:
#   modal app logs devfleet-inference
#   modal app logs devfleet-orchestrator

set -euo pipefail

deploy_inference() {
    echo ">>> Deploying Microservice A (Inference Engine) …"
    modal deploy inference/server.py --detach
    echo ">>> Inference engine deployment started (detached)."
}

deploy_orchestrator() {
    echo ">>> Deploying Microservice B (Orchestrator) …"
    modal deploy orchestrator/core_app.py --detach
    echo ">>> Orchestrator deployment started (detached)."
}

case "${1:-all}" in
    inference)
        deploy_inference
        ;;
    orchestrator)
        deploy_orchestrator
        ;;
    all)
        deploy_inference
        deploy_orchestrator
        echo ""
        echo "Both services deployed.  Retrieve logs with:"
        echo "  modal app logs devfleet-inference"
        echo "  modal app logs devfleet-orchestrator"
        ;;
    *)
        echo "Usage: $0 [inference|orchestrator|all]"
        exit 1
        ;;
esac
