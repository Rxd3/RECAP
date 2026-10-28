# HMM baseline — Member 2 (Raşit)

**Status:** dataset-independent implementation complete; real-data training,
hyperparameter selection, runtime experiments and official scores are pending.
The user reports that the team selected **SponSpeech** and is preparing it.
The exact source URL/revision, format, annotations and split manifest have not
been supplied. No SponSpeech-specific loading or annotation policy is assumed.

## Implemented interface

`punct_restore.models.hmm.HMMModel` provides:

- `HMMModel(label_mode=..., labels=..., smoothing=0.1, min_token_count=2)`;
  inject the ordered vocabulary from `configs/label_sets.yaml`.
- `fit(training_records, metadata=None)` returns the fitted instance and accepts
  only shared-schema records with `split="train"` and the matching mode.
- `predict(records)` returns `sample_id`, `model_name="hmm"`, `label_mode` and
  one `predicted_labels` entry per token, in input sequence order.
- `decode(tokens)` exposes token-sequence Viterbi decoding for adapters/tests.
- `save_model(path)` / `HMMModel.load_model(path)` persist validated JSON counts,
  label order, hyperparameters and JSON provenance. Existing files are not overwritten.
- `sequence_log_probability(tokens, labels)` returns the specified path's joint
  log probability including END; it is not a confidence or marginal likelihood.

Modes A/B use punctuation states (4/8). Mode C uses the 12 standard-punctuation ×
casing states. Training derives joint targets from aligned punctuation/case
arrays. The model never converts B→A/C, retokenizes, lowercases or extracts
labels from raw text. Those policies belong to the approved data release.

## Estimation and decoding

Let `K` be the configured state count, `N` the number of training sequences,
`C(s)` the number of tokens labeled `s`, and `alpha > 0` additive smoothing.

```text
P(start=s) = (start_count(s) + alpha) / (N + alpha*K)
P(next=t | s) = (transition_count(s,t) + alpha) / (C(s) + alpha*(K+1))
P(END | s) = (end_count(s) + alpha) / (C(s) + alpha*(K+1))
P(word=w | s) = (emission_count(s,w) + alpha) / (C(s) + alpha*(|V|+1))
```

Each record is one independent sequence. Counts never cross record boundaries.
END is a separate termination transition, not an extra predicted label. Every
transition row, including END, and every emission row, including UNK, normalizes
to one. Unobserved labels remain possible through smoothing.

This is supervised estimation from gold states, not unsupervised Baum–Welch.
The HMM assumes first-order state dependence and observations independent of
other observations given their states. Viterbi maximizes the complete path using
log probabilities, backpointers and learned END probabilities. It runs in
`O(T*K^2)` time with `O(T*K)` backpointer memory. Exact local ties choose the
earliest predecessor/final state in the supplied vocabulary order. Basic HMM
assumptions and Viterbi follow [Jurafsky and Martin, Appendix A](https://web.stanford.edu/~jurafsky/slp3/A.pdf);
the smoothing, termination and OOV policies above are this baseline's explicit
implementation choices.

## Unknown words and limitations

Vocabulary uses only training word frequencies. Words with frequency below
`min_token_count` are pooled into a per-state UNK observation. At prediction,
every word outside the retained vocabulary uses that same bucket. A literal
token named `<UNK>` remains distinct from the internal bucket. Sparse emission
counts avoid storing a dense vocabulary-by-label matrix; training streams
records and retains counts and the ID-uniqueness set, not full sequences.

Defaults `smoothing=0.1` and `min_token_count=2` are **untuned starting values**.
Select them on validation after the release arrives. With cutoff 1, UNK receives
only smoothing mass if no words were pooled. The baseline cannot distinguish
different unseen words and has no context-feature or neural backoff. It cannot
restore detailed casing outside the three approved classes. Weak future results
must be reported rather than hidden by changing splits or gold labels.

Prediction reads only schema version, IDs and tokens. Gold arrays, original
text and arbitrary metadata are ignored even if accidentally supplied. Invalid
records, duplicate IDs, mismatched lengths, wrong modes and held-out training
records fail explicitly. Artifact loading checks dimensions, nonnegative
counts, incoming/outgoing balance and vocabulary consistency.

## Commands ready for the data handoff

Inspect the CLI now without any dataset:

```powershell
python scripts/train_hmm.py --help
python -m pytest tests/test_hmm.py tests/test_hmm_cli.py -q
```

The following is a **future command template**, not an experiment already run.
Replace all release/version/path placeholders with Member 1's verified values:

```powershell
python scripts/train_hmm.py --train data/processed/RELEASE/A/train.jsonl --output-dir artifacts/models/hmm-A-RUN --label-mode A --smoothing 0.1 --min-token-count 2 --dataset-release-id RELEASE --preprocessing-version VERSION --tokenizer-version VERSION --label-policy-version VERSION --split-manifest data/splits/RELEASE/manifest.json
```

Add `--predict-input path/to/validation.jsonl` to export predictions after fitting.
No evaluation calculations or test-set selection are performed by this script.
For mode C supply C-mode training records; for B supply native extended labels.

Each new run directory contains `model.json` and `run.json`; optional prediction
export adds `predictions.jsonl` and `predictions.manifest.json`. Provenance records
training/manifest/label-config SHA-256 values, release/policy versions, ordered
labels, hyperparameters, deterministic/no-seed status, code commit/dirty flag,
Python/platform, portable command and model/prediction hashes. Training time
includes input reading, validation and estimation. No calibrated confidence or
benchmark latency is claimed. External paths are recorded as filenames plus
hashes to avoid embedding personal absolute paths; relocate inputs when replaying.

The CLI checksums the supplied split manifest but does **not** claim to audit
its sample/group membership, since Member 1 has not delivered that schema yet.
Obtain a validated, duplicate/leakage-audited release before official training.
Any needed SponSpeech adapter must be agreed with Member 1 rather than guessed.

## What the dataset handoff must include

- [ ] Exact source link/revision and verified release ID/usage conditions.
- [ ] Tokenized train/validation/test JSONL or a clear mapping to the shared schema.
- [ ] Stable IDs, fixed split/group manifest and file checksums.
- [ ] Tokenization, lowercasing, punctuation/casing policies and their versions.
- [ ] Available modes and verified casing; an approved projection if any is used.
- [ ] Known exclusions, class distributions and clean/noisy alignment limitations.

Then inspect a small sample, validate compatibility with Member 1, train on train
only, select smoothing/cutoff on validation, freeze settings, and hand prediction
bundles to Member 5 for evaluation and compatible artifacts to Member 6. Do not
add test-driven tuning or replace missing gold casing with guessed labels.

## Completion and evidence

- [x] Supervised counts, smoothing, explicit boundaries and log-space Viterbi.
- [x] Train-only rare/unknown handling and all A/B/C label spaces.
- [x] Shared output alignment, gold-invariance and validated save/load.
- [x] CLI, portable provenance and checksummed optional prediction export.
- [x] Hand-calculated probabilities, exhaustive path checks and temporary CLI tests.
- [ ] Real release inspection, hyperparameter selection, official training and scores.
- [ ] Measured runtime/limitations and evaluated baseline handoff.

Synthetic fixtures are software checks only. No real dataset, trained research
artifact, performance table or experimental result has been committed.

Verification for this increment: `python -m pytest -q` passed **59 tests**.
Black (Python 3.11 target, 100-character lines), Ruff checks for imports/Python
compatibility/common correctness, documentation links and `git diff --check`
also passed. Data/output directories retain only their original empty markers.
