# Scripts

Command-line entry points. Run each one from the repository root; each prints its options
with `--help`. Core logic lives in `src/punct_restore/`; the scripts only parse arguments,
read and write files, and report errors as one line on stderr. Full command sequences are in
[REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).

| Script | Owner | Stage | What it does |
| --- | --- | --- | --- |
| `prepare_data.py` | Member 1 | Data | `extract` recreates the SponSpeech transcripts from pinned byte ranges of the official archive (`--archive` for the full-archive fallback); `build` writes a validated, immutable release with its ID lists. |
| `train_hmm.py` | Member 2 | HMM | Trains the HMM on an explicit `--train-ids` list; writes a JSON artifact and run manifest, and optional predictions (`--predict-input`, `--predict-ids`). `--release-manifest` checks the inputs against the release. |
| `predict_hmm.py` | Member 2 | HMM | Predicts with a saved artifact (`--ids` or `--all-records`), without training. |
| `run_hmm_experiments.py` | Member 2 | HMM | `tune`: frozen validation search from `configs/hmm_sponspeech_v2.yaml`. `finalize`: predicts and scores the selected models on validation, test and test_amb. |
| `verify_hmm_experiments.py` | Member 2 | HMM | Independent audit of a study: score recounts, refits of the winners, prediction comparison. |
| `package_hmm_handoff.py` | Member 2 | HMM | Packages a verified study as a ZIP with internal and outer SHA-256 sums. |
| `train_crf.py` | Member 3 | CRF | `train` fits a linear-chain CRF (`--preset basic` or `improved`) on an explicit `--ids` list and writes a `.joblib` model and manifest. `predict` writes prediction JSONL and, with `--manifest-out`, the prediction manifest. `inspect` prints an artifact's metadata. |
| `run_ablations.py` | Team (final push) | CRF | Trains the improved CRF with one feature group removed (or the filler cue added) on a training sample and writes each variant's primary-validation predictions; score them with `evaluate_models.py`. Never reads test. |
| `tune_threshold.py` | Team (final push) | CRF | Threshold decoding of CRF marginals: `tune` picks τ on validation, `predict` applies the frozen τ once. |
| `evaluate_models.py` | Member 5 | Evaluation | `score` one prediction file, `compare` two on the same IDs (paired bootstrap), `errors` (confusion pairs and examples), `baseline` (the end-of-utterance shortcut). |
| `run_demo.py` | Member 6 | Demo | Loads a saved HMM (`.json`) or CRF (`.joblib`, `.pkl`) through `src/punct_restore/pipeline.py` and prints the restored text; without `--text` it prompts interactively. |

## Interfaces

Flags below are the ones each script accepts (checked with `--help`; see it for defaults).
Every script runs from a plain checkout; each adds `src` to the import path itself.

```text
prepare_data.py extract --raw-root RAW [--archive sponspeech.tar.gz]
prepare_data.py build   --raw-root RAW --release-id ID [--processed-dir DIR] [--splits-dir DIR]

train_hmm.py --train JSONL --train-ids IDS --output-dir DIR [--label-mode {A,B,C}]
    [--smoothing S] [--min-token-count N] --dataset-release-id ID --preprocessing-version V
    --tokenizer-version V --label-policy-version V --split-manifest PATH
    [--predict-input JSONL] [--predict-ids IDS] [--release-manifest PATH]
predict_hmm.py --model PATH --input JSONL --output-dir DIR (--ids IDS | --all-records)
    [--release-manifest PATH] [--selection-record PATH]
run_hmm_experiments.py tune --config YAML --output-dir DIR --evaluator-root DIR
run_hmm_experiments.py finalize --run-dir DIR --evaluator-root DIR
verify_hmm_experiments.py --run-dir DIR --evaluator-root DIR --output-dir DIR
package_hmm_handoff.py --run-dir DIR --evaluator-root DIR --output-dir DIR

train_crf.py train --mode {A,B,C} --input JSONL --ids IDS --output MODEL.joblib
    [--preset {basic,improved}] [--c1 C1] [--c2 C2] [--max-iterations N]
    [--run-id ID] [--manifest-out PATH] [--overwrite]
train_crf.py predict --model MODEL.joblib --input JSONL --output PRED.jsonl [--ids IDS]
    [--manifest-out PATH] [--overwrite]
train_crf.py inspect --model MODEL.joblib
run_ablations.py --mode {A,B,C} --out DIR [--sample 50000] [--variants V1,V2,...]
    [--data-root data] [--release sponspeech-m1-v2]
    variants: improved, improved_minus_{window2,ngrams,question,clause,casing,position},
              improved_plus_filler
tune_threshold.py tune --model M --gold G --ids I [--masks K] [--grid 0.05:0.95:0.05] --out experiments/runs/tau_<mode>.json
tune_threshold.py predict --model M --input JSONL --ids I --tau-file experiments/runs/tau_<mode>.json
    --out PRED --manifest-out MAN

evaluate_models.py score --gold JSONL --ids IDS --predictions PRED [--masks K] --out REPORT.json
    [--manifest MAN] [--bootstrap N] [--seed S]
evaluate_models.py compare --gold JSONL --ids IDS [--masks K] --out REPORT.json
    --predictions-a A --predictions-b B [--bootstrap N] [--seed S]
evaluate_models.py errors --gold JSONL --ids IDS --predictions PRED [--masks K] --out REPORT.json
    [--block NAME] [--per-pair K] [--seed S]
evaluate_models.py baseline --gold JSONL --ids IDS --out PRED.jsonl

run_demo.py --model-path MODEL [--text "some lowercased text"]
```

`--masks` is required in mode C (`masks/case_joint/<split>.jsonl`). `run_demo.py` exits 1
with one friendly line when the model file or the text is unusable. The data, HMM and
evaluation scripts refuse to overwrite existing outputs; `train_crf.py` replaces a file only
with `--overwrite`.
