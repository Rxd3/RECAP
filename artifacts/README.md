# Local artifacts

Owner: Member 6 coordinates storage; each model owner supplies its artifacts.
Everything here except this README is ignored. Do not commit weights, checkpoints, serialized
estimators, or downloaded pretrained models.

Handoffs need a versioned artifact location, SHA-256, model family and
mode, ordered labels, input/preprocessing contract version, feature settings,
training split hash, seed, code commit, dependencies, and load instructions.
The trained models are on the team Drive, not in Git:

- **HMM:** `hmm-sponspeech-v2-20261029.zip` in [`Models/`](https://drive.google.com/drive/folders/13bjpo8GLVOwTJFJoOsG97O_wSyVF5rsN): the three selected models
  (one per mode), the full study and per-file checksums ([HMM_RESULTS.md](../experiments/HMM_RESULTS.md)).
- **CRF:** the six `crf_mode_<a|b|c>_<basic|improved>_full_01.joblib` models with a `SHA256SUMS` file in
  [`RECAP_M3_DRIVE_MODELS`](https://drive.google.com/drive/folders/1jSfe5U8yZJxseU7aA9p6ycQNoZCtx7e_), as shared by Member 3; per-model SHA-256 in
  [CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md) §7.

A local copy of either goes under `artifacts/` (ignored); see [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
Load the selected JSON with `HMMModel.load_model(path)`. The dataset is distributed
separately through the Member 1 release. Keep these generated files outside Git.
