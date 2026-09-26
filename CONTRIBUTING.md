# Contributing

## Branch, push, review

`main` is protected. Nobody pushes to it directly, including maintainers.

```bash
git switch -c your-branch      # branch off main
# ... work ...
git push -u origin your-branch
```

Then open a pull request. It needs:

- **one approving review** — from a code owner (see [.github/CODEOWNERS](.github/CODEOWNERS))
- **green CI** — the `attribution`, `hygiene`, and `checks` jobs
- **all conversations resolved**

Stale approvals are dismissed when new commits land, so a review approves the
code that actually merges. History on `main` is linear: rebase your branch
rather than merging `main` into it.

Force-pushing to `main` and deleting `main` are blocked outright.

## Attribution: no AI tools as contributors

Use whatever tools you like — editors, copilots, agents. They are instruments,
and they do not get a line in this repo's contributor graph. Commit under your
own name, and do not let a tool add itself as a co-author.

Concretely, these are rejected:

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-authored-by: Copilot <copilot@users.noreply.github.com>
🤖 Generated with [Claude Code](...)
Generated with Cursor
```

This is enforced in two places, and checks the commit **author**, the
**committer**, and the message **trailers**:

| Where | What |
| --- | --- |
| Your machine | `commit-msg` and `pre-push` hooks in [.githooks/](.githooks/) |
| GitHub | the `attribution` job in CI, on every push and pull request |

The hooks are optional and CI is not, so a missing hook costs you a red build
rather than a bad commit on `main`.

Mentioning these systems in prose is completely fine — this project is *about*
them, and the check is deliberately narrow enough to allow it. Only credit
lines and author identities are scanned.

### Install the hooks

```bash
bash scripts/install-hooks.sh        # macOS, Linux, Git Bash, WSL
```

```powershell
pwsh scripts/install-hooks.ps1       # Windows PowerShell
```

Both just set `core.hooksPath` to `.githooks`, so the hooks stay version
controlled and update when you pull.

### Turning off a tool's automatic trailer

Most tools that add attribution have a setting for it. If yours does not, strip
it before pushing:

```bash
git commit --amend        # delete the line, save, exit
git rebase -i <base>      # for several commits: mark them 'reword'
```

### Running the check yourself

```bash
bash scripts/check-ai-attribution.sh --identity              # your git config
bash scripts/check-ai-attribution.sh --range main..HEAD      # your branch
bash scripts/test-check-ai-attribution.sh                    # the check's own tests
```

To cover a tool the list does not know about yet, add it to `AI_PATTERN` in
[scripts/check-ai-attribution.sh](scripts/check-ai-attribution.sh) and add a
case to the test file in the same pull request.

## CI

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs three jobs:

- **attribution** — the rule above, plus its own self-tests
- **hygiene** — shellcheck, no tracked file over 5 MB, nothing committed that
  `.gitignore` excludes
- **checks** — Python lint and tests on Linux, macOS and Windows. There is no
  Python yet, so it reports "nothing to lint or test" until a `pyproject.toml`
  lands, at which point it starts running for real with no edits needed.
