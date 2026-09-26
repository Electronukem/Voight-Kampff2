#!/usr/bin/env bash
#
# Tests for check-ai-attribution.sh. Run locally or in CI:
#   bash scripts/test-check-ai-attribution.sh
#
# The important cases are the negative ones. This project writes about Claude,
# GPT, Gemini and friends by name constantly -- the check must flag attribution
# lines without flagging ordinary prose that mentions a model.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TD="$(mktemp -d)"
trap 'rm -rf "$TD"' EXIT
pass=0; fail=0

check() { # <name> <expected-exit> <message>
  printf '%s\n' "$3" > "$TD/m.txt"
  out="$(bash scripts/check-ai-attribution.sh --message "$TD/m.txt" 2>&1)"; rc=$?
  if [ "$rc" = "$2" ]; then
    pass=$((pass + 1)); printf 'PASS  %s\n' "$1"
  else
    fail=$((fail + 1)); printf 'FAIL  %s (expected exit %s, got %s)\n' "$1" "$2" "$rc"
    printf '%s\n' "$out" | sed 's/^/      /' | head -5
  fi
}

# --- must be rejected ---------------------------------------------------
check "Claude co-author trailer" 1 "Add thing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
check "Copilot co-author trailer" 1 "Fix bug

Co-authored-by: Copilot <copilot@users.noreply.github.com>"
check "lowercase trailer" 1 "Fix bug

co-authored-by: claude <x@example.com>"
check "robot footer" 1 "Add docs

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
check "generated-with footer" 1 "Update deps

Generated with Cursor"
check "signed-off-by a bot" 1 "Fix

Signed-off-by: Anthropic Bot <bot@anthropic.com>"

# --- must be allowed ----------------------------------------------------
check "plain message" 0 "Add scenario loader"
check "human co-author" 0 "Add scenario loader

Co-authored-by: Ada Lovelace <ada@example.com>"
check "prose naming models" 0 "Add Claude Fable 5.1 and GPT-6 Astra to the policy table

The RoboHarm results from the Anthropic and OpenAI policies need a row
each. Cursor position in the table matters for readability."

printf -- '---\npassed=%s failed=%s\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
