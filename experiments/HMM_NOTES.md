# HMM baseline — Member 2 (Raşit)

**Status:** dataset-independent implementation complete; real-data training,
hyperparameter selection, runtime experiments and official scores are pending.
Member 1's corrected release **sponspeech-m1-v2** (policy **m1-v2**) is verified;
see [release validation](../docs/DATA_RELEASE_VALIDATION.md) and
[data usage](../data/README.md). Compatibility-only fits/predictions in all three
modes pass. No full-data model or benchmark score has been produced.

The release-aware CLI now requires frozen training IDs and verifies the release
manifest, versions and label configuration. The experiment and independent
reproduction commands are in [HMM_RUNBOOK.md](HMM_RUNBOOK.md). Full-data runs and
observed results remain pending until those commands finish.

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

## Experiment workflow and completion evidence

Use [HMM_RUNBOOK.md](HMM_RUNBOOK.md) and the predeclared configuration
[configs/hmm_sponspeech_v2.yaml](../configs/hmm_sponspeech_v2.yaml). The shared
Member 5 evaluator is pinned at commit 3044a58e777200bcac64e6e3364922404be359bf.

- Training requires explicit IDs; missing/duplicate IDs and invalid unselected
  records fail before output. Source-file order is preserved.
- The release manifest authenticates canonical files, default-training IDs,
  label configuration, source revision and policy versions.
- The predictor strips gold/source metadata and loads the saved artifact.
- Search is validation-only; all modes freeze before held-out evaluation.
- The audit independently checks metrics and performs full refit/prediction
  reproduction. Generated models and predictions remain outside Git.

Full-data selection, measured limitations and evaluated handoff are pending at
this implementation checkpoint. No benchmark numbers are claimed here.
