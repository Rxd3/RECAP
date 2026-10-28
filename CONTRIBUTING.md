# Contributing

Work on your assigned member branch and open focused pull requests into `main`.
Use [TEAM_TASKS.md](TEAM_TASKS.md) to find owners/reviewers and
[docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for detailed beginner commands.
Current code is intentionally unfinished; a placeholder is not a working model.

## Daily workflow

1. Read the shared data/model/evaluation contracts and relevant open decisions.
2. Check `git status`, switch to your branch, then merge updated `main` regularly.
3. Edit primarily your assigned files. Discuss shared interfaces before coding
   and ask affected owners to review coordinated changes.
4. Keep logic in `src/punct_restore/` and future scripts thin. Put adjustable
   settings in `configs/` and use repository-relative paths.
5. Add small meaningful tests with actual behavior. Tests must not download data,
   contact live APIs, require pretrained weights or need a GPU.
6. Inspect `git diff`, stage named files, inspect `git diff --cached`, then commit.
7. Push only to the team's Project 2 remote after it has been provided. Open a
   PR with problem/behavior, affected contracts, checks and pending work.
8. Obtain review before merging into `main`. Do not force-push another member's
   branch or make direct feature-development commits to `main`.

## Environment and validation

Follow the README environment setup. Runtime dependencies are planned, not yet
installed or pinned for experiments. Add only needed dependencies with a clear
reason and a compatible tested environment in implementation PRs.

The three `tests/test_*_contract.py` files contain TODO plans, not real tests.
Do not call a zero-test collection a successful functional test. Until behavior
exists, safe checks are Python syntax parsing, TOML/YAML parsing, label counts,
file/link structure and ignore rules, as recorded in
[initial verification](docs/INITIAL_VERIFICATION.md). Once genuine tests exist:

```powershell
python -m pytest
```

No training, feature extraction, inference or scoring is expected to work today.
The six task scripts intentionally return a nonzero exit with an unfinished message.

## Data, artifacts and claims

Keep downloaded/processed data, model weights, prediction dumps, generated
outputs, credentials, `.env` files and personal paths out of commits. Directory
markers and documentation remain tracked. Never use `git add -f` to bypass these
rules. Share versioned data/artifacts through an agreed location with checksums.

Any annotation, split, target projection, feature interface or metric change
requires affected owners' review and a versioned decision. Preserve gold-free
prediction inputs. Never fabricate scores, copy gold into predictions, or tune
against final test data. Record limitations and negative findings.

## Pull-request checklist

- [ ] Scope and ownership are clear; shared changes were discussed first.
- [ ] Contracts/configuration/documentation match the implemented behavior.
- [ ] Relevant meaningful checks passed; skipped/pending checks are explained.
- [ ] Staged diff contains only intended files; no data/artifact/secret leakage.
- [ ] Data, mode, policy and artifact compatibility changes are documented.
- [ ] Reproduction commands and real evidence accompany experimental claims.
