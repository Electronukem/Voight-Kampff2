# Point git at the repo's tracked hooks. Run once after cloning.
# Windows PowerShell equivalent of install-hooks.sh.
# The hooks are bash scripts; Git for Windows ships the bash they need.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
git config core.hooksPath .githooks
Write-Host "hooks enabled: core.hooksPath = .githooks"
bash scripts/check-ai-attribution.sh --identity
