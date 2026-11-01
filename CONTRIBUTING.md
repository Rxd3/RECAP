# Contributing

Work on your assigned member branch and open focused pull requests into `main`.
Use [TEAM_TASKS.md](TEAM_TASKS.md) to find owners and reviewers.

## Daily workflow

1. Read the shared data/model/evaluation contracts and relevant open decisions.
2. Check `git status`, switch to your branch, then merge updated `main` regularly.
3. Edit primarily your assigned files. Discuss shared interfaces before coding
   and ask affected owners to review coordinated changes.
4. Keep logic in `src/punct_restore/`. A new command is a module in `src/punct_restore/cli/` (or
   `src/punct_restore/reporting/`) with a docstring and `main(argv)`, listed in `COMMANDS`; `scripts/`
   holds only 3-line stubs. Put adjustable settings in `configs/` and use repository-relative paths.
5. Add small meaningful tests with actual behavior. Tests must not download data,
   contact live APIs, require pretrained weights or need a GPU.
6. Inspect `git diff`, stage named files, inspect `git diff --cached`, then commit.
7. Push to the team remote (`https://github.com/Rxd3/RECAP.git`). Open a PR with the
   problem or behaviour, affected contracts, checks run and open work.
8. Obtain review before merging into `main`. Do not force-push another member's
   branch or make direct feature-development commits to `main`.

## Environment and validation

Follow the README environment setup. Install the package in editable mode (`pip install -e ".[dev]"`):
`recap` and the stubs resolve `configs/` and their default folders through the checkout, so a plain
`pip install .` does not work. The completed HMM study pins PyYAML in
`requirements-hmm.txt` and records its tested Python version. Add only needed
dependencies with a clear reason and a compatible tested environment in
implementation PRs.

The whole suite runs with pytest from the repository root, on small fixtures only, and the code must
pass the linter (ruff 0.17.3 from the `dev` extra; settings in `pyproject.toml`). Both must pass before
you open a pull request and again before it is merged:

```powershell
python -m pytest -q
ruff check src scripts tests
```

The CI workflow `.github/workflows/ci.yml` runs the same two checks on Python 3.12 for every push and
pull request.

- **One branch per member:** work on your `feat/m<N>-...` branch (owners and branches are in
  [TEAM_TASKS.md](TEAM_TASKS.md)), not on `main`.
- **Conventional commit subjects:** `type(scope): summary`, for example
  `feat(data): add verified preprocessing pipeline` or `fix(eval): ...`, `docs: ...`, `test: ...`,
  `chore: ...`.
- **Stage named files, check, then commit:** `git status`, `git add <paths>` (not `git add .`),
  `git diff --cached`, `git commit`. Merge `main` into your branch regularly and never force-push
  another member's branch.
- **Merge through pull requests, as merge commits,** so every member keeps their own commits. The
  integration branch `release/final` merged the member branches in this order: M5 evaluation,
  M2 HMM, M3 CRF, M6 integration (`git log --merges release/final`).

Completed results and how to rerun them: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md),
[HMM results](experiments/HMM_RESULTS.md) and [CRF results](docs/EVALUATION_RESULTS_CRF.md).

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
