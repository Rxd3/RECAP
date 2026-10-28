# Git workflow for six members

`main` is stable integration. Each member submits PRs from their dedicated branch:

| Member | Branch |
| --- | --- |
| 1 | `feat/m1-data-preprocessing` |
| 2 | `feat/m2-hmm-baseline` |
| 3 | `feat/m3-crf-model` |
| 4 | `feat/m4-feature-engineering` |
| 5 | `feat/m5-evaluation` |
| 6 | `feat/m6-integration-presentation` |

All six start at the same initial scaffolding commit. The local setup leaves
`main` checked out. No remote is configured or pushed by scaffolding. The URL
below is a placeholder for the future **Project 2** repository, never NALTRA.

## 1. Clone after Project 2 has been published

Replace `TEAM/PROJECT2` with the actual repository path. Do not run placeholder
commands unchanged. A teammate/repository owner must separately create and
publish the authorized Project 2 remote before others can clone it.

```powershell
git clone https://github.com/TEAM/PROJECT2.git nlp-project2
cd nlp-project2
git remote -v
git status
git fetch origin
git switch --track origin/feat/m1-data-preprocessing
```

Use your own branch from the table. If already working in the scaffold checkout,
the branches exist locally; use `git switch feat/m1-data-preprocessing` instead.
`git remote -v` should identify only the intended Project 2 repository. If an
expected remote branch is absent, ask the owner to publish it; do not guess a base.

## 2. Update before working

Start with a clean working tree; commit intended work first. In the following,
replace Member 1's branch with your own:

```powershell
git status
git switch main
git pull --ff-only origin main
git switch feat/m1-data-preprocessing
git merge main
```

Before a remote exists, skip `pull` and merge available local `main` instead.
Merge updated `main` frequently and after dependency PRs land. Merging preserves
history; do not force-push or rewrite another member's commits.

## 3. Make and review a focused commit

Example for Member 1 after implementing real data behavior:

```powershell
git status --short
git diff -- src/punct_restore/data/ scripts/prepare_data.py docs/DATASET_PLAN.md
git add src/punct_restore/data/ scripts/prepare_data.py docs/DATASET_PLAN.md
git diff --cached --stat
git diff --cached
git commit -m "feat(data): add verified preprocessing pipeline"
```

Stage explicit paths rather than `git add .` for ordinary member work. If an
unrelated file was staged, `git restore --staged path/to/file` removes it from
the staging area while keeping the working copy. Do not discard another
member's uncommitted edits. Use `git diff` to determine who should own a change.

## 4. Push and open a pull request

Only after the correct Project 2 remote exists and team publishing is authorized:

```powershell
git push -u origin feat/m1-data-preprocessing
```

On GitHub, select **Pull requests → New pull request**, base `main`, compare your
member branch. Describe the problem and resulting behavior, tests/checks,
contract/config changes, dependencies, evidence and unresolved work. Request
review from relevant owners. A draft PR can expose interface work early.
Merge only after review; do not push feature work directly to `main`.

## 5. Handle merge conflicts

If `git merge main` reports a conflict:

```powershell
git status
git diff --name-only --diff-filter=U
```

Open each conflicted file, discuss overlapping interface/ownership changes,
combine the intended content and remove `<<<<<<<`, `=======`, `>>>>>>>` markers.
Do not blindly accept one side. Review/run relevant checks, then stage resolved
paths and finish the merge:

```powershell
git add path/to/resolved_file
git commit
```

If unsure how to resolve, `git merge --abort` returns to the pre-merge state.
Start merges with a clean tree so local work is preserved predictably. Ask the
affected owner for help instead of using destructive reset commands.

## 6. First PR dependency order

1. Member 1 proposes sources/annotation/split decisions; Member 5 reviews metrics
   and projection compatibility. Member 6 coordinates contract versioning.
2. Members 3–4 agree feature injection in one focused PR. Member 6 defines
   artifact/CLI integration requirements. These planning PRs can proceed together.
3. Member 1 merges a validated data pipeline/release description and tiny fixtures.
   All consumers merge updated `main` and use the exact same release/split hashes.
4. Members 2 and 3 implement HMM/basic CRF in separate PRs; Member 5 implements
   evaluation against hand-checkable fixtures; Member 6 implements adapters.
5. Member 4 adds improved features/ablations through the stable CRF interface.
6. Freeze settings, evaluate the final test, then integrate verified evidence/demo.

Keep PRs small. A branch may have multiple PRs over the project. Merge accepted
shared interfaces before dependent implementations to limit conflicts.

## Local bootstrap and identity fallback

For a new independent empty directory only, the bootstrap procedure is:

```powershell
git init -b main
git config --get user.name
git config --get user.email
```

If identity is absent, configure **your own** real identity; never invent a team
identity. The following values are placeholders to replace:

```powershell
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
```

After reviewing the generated scaffold and confirming identity:

```powershell
git add .
git diff --cached --stat
git commit -m "chore: initialize sequence labeling project scaffold"
git branch feat/m1-data-preprocessing main
git branch feat/m2-hmm-baseline main
git branch feat/m3-crf-model main
git branch feat/m4-feature-engineering main
git branch feat/m5-evaluation main
git branch feat/m6-integration-presentation main
git switch main
```

These commands describe initial setup, which should be run once; do not
reinitialize an existing/unrelated repository or recreate existing branches.
The actual outcome for this workspace is in
[INITIAL_VERIFICATION.md](INITIAL_VERIFICATION.md). Remote creation, `remote add`
and publication are separate owner actions and were not performed here.
