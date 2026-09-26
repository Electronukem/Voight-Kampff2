#!/usr/bin/env bash
#
# Reject commits that credit an AI tool as a contributor.
#
# This repo attributes work to people. AI tools are instruments, like an editor
# or a compiler, and do not get a line in the contributor graph. GitHub builds
# that graph from three places, so we check all three:
#
#   1. the commit author  (name + email)
#   2. the commit committer (name + email)
#   3. `Co-authored-by:` and friends in the commit message
#
# Used by .githooks/commit-msg (locally) and by .github/workflows/ci.yml (on
# every push and pull request), so local and remote behaviour cannot drift.
#
# Usage:
#   check-ai-attribution.sh --range <rev-list args>  # scan a range of commits
#   check-ai-attribution.sh --message <file>         # scan one message file
#   check-ai-attribution.sh --identity               # scan configured identity
#   check-ai-attribution.sh --text                   # scan stdin (PR title/body)

set -euo pipefail

# Add new tools here. Matched case-insensitively, as an extended regex.
# Keep these specific: they are matched against attribution lines and author
# identities only, never against ordinary prose, because this project writes
# about these systems by name all the time.
AI_PATTERN='claude|anthropic|copilot|codex|chatgpt|openai|gpt-[0-9]|gemini|bard|llama|devin|windsurf|codeium|cursor\.(com|sh)|cursoragent|aider|sweep\.dev|tabnine|sourcegraph|amp-?code|jules|replit[ -]?agent|bolt\.new|lovable\.dev|noreply@anthropic\.com'

# Lines that assign credit. Only these get scanned for tool names.
TRAILER_PATTERN='^[[:space:]]*(co-authored-by|signed-off-by|assisted-by|reviewed-by|helped-by|generated-by|authored-by|on-behalf-of|committed-by)[[:space:]]*:'

# Marketing footers that AI tools append. Flagged wherever they appear.
FOOTER_PATTERN='(generated (with|by) .*(claude|copilot|codex|cursor|gemini|chatgpt|devin))|(🤖 generated)|(co-?authored.?by.?claude)'

fail() {
  # GitHub renders ::error:: as an annotation; locally it is just noise.
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::error::$1" >&2
  fi
  echo "" >&2
  echo "  $1" >&2
  shift
  for line in "$@"; do echo "  $line" >&2; done
  echo "" >&2
  echo "  This repo does not credit AI tools as contributors. Use the tool all" >&2
  echo "  you like -- just commit under your own name, with no tool trailer." >&2
  echo "" >&2
  echo "  To fix the most recent commit:" >&2
  echo "    git commit --amend        # delete the offending line, save, exit" >&2
  echo "" >&2
  echo "  For several commits:" >&2
  echo "    git rebase -i <base>      # mark them 'reword'" >&2
  echo "" >&2
  echo "  If a tool adds the trailer for you automatically, turn that off." >&2
  echo "  See CONTRIBUTING.md." >&2
  exit 1
}

# Scan a commit message body for credit lines and footers naming an AI tool.
scan_message() {
  local body="$1" context="$2" hit

  hit="$(printf '%s\n' "$body" \
    | grep -i -E "$TRAILER_PATTERN" \
    | grep -i -E "$AI_PATTERN" || true)"
  [ -n "$hit" ] && fail "AI tool credited as a contributor in $context:" "$hit"

  hit="$(printf '%s\n' "$body" | grep -i -E "$FOOTER_PATTERN" || true)"
  [ -n "$hit" ] && fail "AI tool attribution footer in $context:" "$hit"

  return 0
}

# Scan an author/committer identity string ("Name <email>").
scan_identity() {
  local ident="$1" context="$2"
  if printf '%s' "$ident" | grep -q -i -E "$AI_PATTERN"; then
    fail "AI tool set as the $context:" "$ident"
  fi
}

mode="${1:---identity}"

case "$mode" in
  --range)
    # Everything after --range is handed to git rev-list verbatim, so callers
    # can pass "a..b" or "<sha> --not --remotes" for a brand new branch.
    shift
    if [ $# -eq 0 ]; then
      echo "usage: --range <rev-list args>" >&2
      exit 2
    fi
    count=0
    while IFS= read -r sha; do
      [ -z "$sha" ] && continue
      count=$((count + 1))
      scan_identity "$(git log -1 --format='%an <%ae>' "$sha")" "author of $(git log -1 --format='%h' "$sha")"
      scan_identity "$(git log -1 --format='%cn <%ce>' "$sha")" "committer of $(git log -1 --format='%h' "$sha")"
      scan_message "$(git log -1 --format='%B' "$sha")" "$(git log -1 --format='%h (%s)' "$sha")"
    done < <(git rev-list "$@")
    echo "ok: $count commit(s) carry no AI attribution"
    ;;

  --message)
    file="${2:?usage: --message <file>}"
    scan_message "$(cat "$file")" "this commit message"
    scan_identity "$(git var GIT_AUTHOR_IDENT | sed 's/> .*/>/')" "commit author"
    scan_identity "$(git var GIT_COMMITTER_IDENT | sed 's/> .*/>/')" "committer"
    ;;

  --identity)
    scan_identity "$(git config user.name || true) <$(git config user.email || true)>" "configured git identity"
    echo "ok: git identity carries no AI attribution"
    ;;

  --text)
    scan_message "$(cat)" "the pull request title/body"
    echo "ok: pull request text carries no AI attribution"
    ;;

  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
