#!/usr/bin/env bash
#
# Tests for check-secrets.sh. The negative cases matter as much as the
# positive ones: a scanner that flags documentation gets disabled, and a
# disabled scanner protects nothing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Fixtures are assembled from fragments at runtime so no complete credential
# pattern ever appears literally in this file. Allowlisting this path in the
# scanner would have been simpler and would also have been a hole: a path the
# scanner skips is a path a real key can hide in.
SK="sk-"; ANT="ant-"; PROJ="proj-"; AK="AKIA"; GH="ghp_"; AIZ="AIza"; XOX="xoxb-"
BEGINPK="-----BEGIN"; RSAPK="RSA PRIVATE KEY-----"
BODY="abcdefghijklmnopqrstuvwxyz0123456789"
BODY32="A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"

TD="$(mktemp -d)"
trap 'rm -rf "$TD"' EXIT
pass=0; fail=0

check() { # <name> <expected-exit> <content>
  printf '%s\n' "$3" > "$TD/f.txt"
  out="$(bash scripts/check-secrets.sh --file "$TD/f.txt" 2>&1)"; rc=$?
  if [ "$rc" = "$2" ]; then
    pass=$((pass + 1)); printf 'PASS  %s\n' "$1"
  else
    fail=$((fail + 1)); printf 'FAIL  %s (expected %s, got %s)\n' "$1" "$2" "$rc"
    printf '%s\n' "$out" | sed 's/^/      /' | head -3
  fi
}

# --- must be rejected ---------------------------------------------------
check "OpenAI sk- key"        1 "OPENAI_API_KEY=${SK}${BODY32}"
check "OpenAI project key"    1 "key = ${SK}${PROJ}${BODY}"
check "Anthropic key"         1 "${SK}${ANT}api03-${BODY}"
check "AWS access key id"     1 "aws_access_key_id = ${AK}2J7Q4XZ9RLMN3BVC"
check "AWS docs example key"  0 "aws_access_key_id = ${AK}IOSFODNN7EXAMPLE"
check "GitHub token"          1 "${GH}${BODY}"
check "Google API key"        1 "${AIZ}SyA0123456789abcdefghijklmnopqrstuvw"
check "Slack token"           1 "${XOX}123456789012-abcdefghijklmnop"
check "private key block"     1 "${BEGINPK} ${RSAPK}"
ANTKEY="ANTHROPIC_API"; HEXBODY="9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c"
check "env assignment"        1 "${ANTKEY}_KEY: ${HEXBODY}"

# --- must be allowed ----------------------------------------------------
check "env var reference"     0 'export OPENAI_API_KEY="$OPENAI_API_KEY"'
check "shell placeholder"     0 'OPENAI_API_KEY=${OPENAI_API_KEY}'
check "docs placeholder"      0 'Set OPENAI_API_KEY=your-key-here before running'
check "angle placeholder"     0 'OPENAI_API_KEY=<your-api-key>'
check "elided key"            0 'PowerShell:  $env:OPENAI_API_KEY = "sk-..."' 
check "plain prose"           0 'The probe needs an API key to evaluate a real model.'
check "a sha256 digest"       0 '04d617c485e59965c65549da600667f10d18a0fd1e97e2250bda0c948da1d12e'
check "python attribute"      0 'api_key_env: str = "ANTHROPIC_API_KEY"'

printf -- '---\npassed=%s failed=%s\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
