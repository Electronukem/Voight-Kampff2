#!/usr/bin/env bash
#
# Reject anything that looks like an API key.
#
# Evaluating real models needs a key. A key committed to a public repository is
# compromised within minutes -- crawlers watch the GitHub event firehose for
# exactly this -- and rotating it is the *only* remedy, because rewriting
# history does not un-publish what was already fetched.
#
# So the key lives in the environment and nowhere else. This checks that.
#
# Used by .githooks/pre-commit (before it is written), .githooks/pre-push
# (before it leaves the machine), and CI (because hooks are opt-in).
#
# Usage:
#   check-secrets.sh --staged            # what is about to be committed
#   check-secrets.sh --range <revs>      # commits about to be pushed
#   check-secrets.sh --tree              # every tracked file
#   check-secrets.sh --file <path>       # one file

set -uo pipefail

# Vendor key formats, matched on their prefixes and minimum lengths. Prefixes
# rather than entropy heuristics: entropy flags base64 blobs and test
# fixtures, and a benchmark repo is full of hashes.
readonly PATTERNS=(
  'sk-ant-[A-Za-z0-9_-]{20,}'
  'sk-proj-[A-Za-z0-9_-]{20,}'
  'sk-[A-Za-z0-9]{32,}'
  'AKIA[0-9A-Z]{16}'
  'gh[pousr]_[A-Za-z0-9]{30,}'
  'AIza[0-9A-Za-z_-]{30,}'
  'xox[baprs]-[A-Za-z0-9-]{10,}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
)

# An assignment with a real-looking value. Placeholders are allowed, because
# documentation has to show the shape of the thing.
readonly ASSIGNMENT='(OPENAI|ANTHROPIC|HUGGINGFACE|HF|AWS_SECRET|GOOGLE|GEMINI)[A-Z_]*(KEY|TOKEN|SECRET)[[:space:]]*[=:][[:space:]]*["'"'"']?[A-Za-z0-9_+/-]{16,}'

# Text that is obviously illustrative rather than a live credential.
readonly PLACEHOLDER='(your[-_]?|my[-_]?|example|placeholder|dummy|fake|test[-_]?key|xxx+|\.\.\.|<[a-z-]+>|\$\{|\$[A-Z_]+|sk-\.\.\.|REDACTED|CHANGEME)'

report() {
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::error::$1" >&2
  fi
  echo "" >&2
  echo "  $1" >&2
  shift
  for line in "$@"; do echo "  $line" >&2; done
  cat >&2 <<'HELP'

  Nothing that looks like a credential may enter this repository.

  Keep the key in your environment instead:

    PowerShell   $env:OPENAI_API_KEY = "sk-..."
    bash/zsh     export OPENAI_API_KEY="sk-..."

  For something persistent, put it in a .env file -- .env is gitignored and
  must stay that way.

  If a key HAS already been committed and pushed, treat it as compromised and
  rotate it now. Rewriting history does not help: anything public was very
  likely already fetched.
HELP
  exit 1
}

scan_text() {
  local text="$1" context="$2" hit
  for pattern in "${PATTERNS[@]}"; do
    # -e is required: several patterns begin with '-' and would otherwise be
    # parsed as grep options rather than as the thing being searched for.
    hit="$(printf '%s\n' "$text" | grep -nE -e "$pattern" | grep -vEi -e "$PLACEHOLDER" || true)"
    if [ -n "$hit" ]; then
      report "possible API key in $context:" "$(printf '%s' "$hit" | cut -c1-120)"
    fi
  done
  hit="$(printf '%s\n' "$text" | grep -nEi "$ASSIGNMENT" | grep -vEi "$PLACEHOLDER" || true)"
  if [ -n "$hit" ]; then
    report "credential assigned a literal value in $context:" "$(printf '%s' "$hit" | cut -c1-120)"
  fi
}

mode="${1:---staged}"

case "$mode" in
  --staged)
    files="$(git diff --cached --name-only --diff-filter=ACMR)"
    [ -z "$files" ] && { echo "ok: nothing staged"; exit 0; }
    count=0
    while IFS= read -r file; do
      [ -z "$file" ] && continue
      git check-attr binary -- "$file" | grep -q 'binary: set' && continue
      scan_text "$(git show ":$file" 2>/dev/null || true)" "$file"
      count=$((count + 1))
    done <<< "$files"
    echo "ok: no credentials in $count staged file(s)"
    ;;

  --range)
    shift
    [ $# -eq 0 ] && { echo "usage: --range <rev-list args>" >&2; exit 2; }
    count=0
    while IFS= read -r sha; do
      [ -z "$sha" ] && continue
      count=$((count + 1))
      scan_text "$(git show --no-color --format= "$sha" 2>/dev/null || true)" "commit $(git log -1 --format='%h (%s)' "$sha")"
    done < <(git rev-list "$@")
    echo "ok: no credentials in $count commit(s)"
    ;;

  --tree)
    count=0
    while IFS= read -r file; do
      [ -z "$file" ] && continue
      git check-attr binary -- "$file" | grep -q 'binary: set' && continue
      scan_text "$(cat "$file" 2>/dev/null || true)" "$file"
      count=$((count + 1))
    done < <(git ls-files)
    echo "ok: no credentials in $count tracked file(s)"
    ;;

  --file)
    scan_text "$(cat "${2:?usage: --file <path>}")" "$2"
    echo "ok: no credentials in $2"
    ;;

  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
