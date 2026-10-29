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

Follow the README environment setup. The completed HMM study pins PyYAML in
`requirements-hmm.txt` and records its tested Python version. Add only needed
dependencies with a clear reason and a compatible tested environment in
implementation PRs.

Member 2's synthetic HMM and CLI tests now run with pytest. The three
`tests/test_*_contract.py` files still contain shared TODO plans. For unfinished
components, use structural/configuration checks without claiming functional
coverage. Original checks are recorded in
[initial verification](docs/INITIAL_VERIFICATION.md). Run implemented tests with:

```powershell
python -m pytest
```

The HMM's full-data training, validation selection, saved-model inference and
independent reproduction are complete; see [results](experiments/HMM_RESULTS.md)
and [runbook](experiments/HMM_RUNBOOK.md). Its study uses the pinned Member 5
scorer in a separate checkout. CRF, ablations and the integrated text-decoding
workflow remain unfinished on this branch.

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
