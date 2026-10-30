# Local artifacts

Owner: Member 6 coordinates storage; each model owner supplies its artifacts.
`models/` is empty and ignored apart from its directory marker. Do not commit
weights, checkpoints, serialized estimators, or downloaded pretrained models.

Handoffs need a versioned artifact location, SHA-256, model family and
mode, ordered labels, input/preprocessing contract version, feature settings,
training split hash, seed, code commit, dependencies, and load instructions.
HMM model I/O is implemented. Member 2's local handoff is
`hmm/hmm-sponspeech-v2-20261029.zip`; its selected-model index, full study and
per-file checksums are described in [HMM_RESULTS.md](../experiments/HMM_RESULTS.md).
Load the selected JSON with `HMMModel.load_model(path)`. The dataset is distributed
separately through the Member 1 release. Keep these generated files outside Git.
