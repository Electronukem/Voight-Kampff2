#!/usr/bin/env bash
# Point git at the repo's tracked hooks. Run once after cloning.
# Works on macOS, Linux, and Windows (Git Bash / WSL).
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true
echo "hooks enabled: core.hooksPath = .githooks"
bash scripts/check-ai-attribution.sh --identity
