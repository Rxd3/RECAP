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
| `project_joint.py` | Team (final push) | Study (H1, B→A) | Projects mode C (`--from C`, default; `model_name` `<name>_joint_C`) or mode B predictions (`--from B`, the approved D5 mapping; `<name>_from_B`) onto mode A punctuation labels, with a manifest. |
| `build_leakage_subsets.py` | Team (final push) | Study (leakage) | Writes `all.ids`, `kept.ids`, `excluded_train_video.ids`, `excluded_other_leak.ids`, `excluded_quality.ids` and `counts.json` for one split and mode. |
| `reporting/build_m3_tables.py` | Team (final push) | Reporting | CRF result tables for one split from `results/scores/`; `--tau` adds the threshold rows (test only). |
| `reporting/build_m3_final_sections.py` | Team (final push) | Reporting | Threshold, leakage, test_amb and runtime sections from `results/scores/` and the training manifests (`--manifests`, default `<scores-root>/crf_training_manifests`). |
| `reporting/build_ablation_table.py` | Team (final push) | Reporting | Feature-ablation tables from `results/scores/ablations/`. |
| `reporting/time_models.py` | Team (final push) | Reporting | Median-of-3 prediction timing after a 200-record warm-up (and an optional HMM fit). |
| `reporting/build_tables.py`, `reporting/_cli.py` | Team (final push) | Reporting | Shared helpers (half-up percentages, cells, JSON loading) and one-line error handling; not run directly. |
| `train_hidden_event_lm.py` | Team (final push) | Comparison | `train` fits a Stolcke-style hidden-event LM (mode C adds a unigram truecaser); `predict` writes predictions and, with `--manifest-out`, a manifest. |
| `factor_predictions.py` | Team (final push) | Comparison | Joins mode A predictions and CASE predictions (a casing-only CRF) into mode C predictions with a manifest. |
| `simulate_asr_noise.py` | Team (final push) | Robustness | Copies the validation or test inputs of every mode with simulated ASR word errors (`--edit-model {v1,v2}`: `v1` has deletion, homophone substitution, adjacent swap; `v2` adds `default_train` confusion substitution, filler/function-word insertion, <=3-char deletion, and 10% swap cap) at a target WER; labels and casing masks stay aligned. |
| `find_gold_noise.py` | Team (final push) | Study (gold noise) | Flags speaker labels, subtitle cue residue and auto-caption videos on the primary mode B lists; writes `<split>_B_noisy.tsv` and `<split>_B_summary.json`. |
| `reporting/asr_noise_table.py` | Team (final push) | Reporting | Builds the multi-seed ASR noise `v2` summary JSON and `docs/ASR_NOISE_V2.md` from `results/scores/`. |
| `reporting/asr_noise_stats.py`, `reporting/asr_noise_report.py`, `reporting/asr_noise_findings.py` | Team (final push) | Reporting (modules) | Helpers of `asr_noise_table.py`: exact seed statistics and single rounding, the report's Markdown, and the findings computed from the summary. Not run on their own. |
| `reporting/crf_convergence.py` | Team (final push) | Reporting | Loss-trace summary of trusted CRF `.joblib` models. |
| `reporting/length_position.py` | Team (final push) | Reporting | Mode A micro F1 by utterance length and QUESTION recall by position (EVALUATION_RESULTS.md §6). |
| `run_demo.py` | Member 6 | Demo | Loads a saved HMM or hidden-event LM (`.json`), or a CRF or gap classifier (`.joblib`, `.pkl`), through `src/punct_restore/pipeline.py` (CASE-mode CRFs are refused) and prints the restored text; without `--text` it prompts interactively. |

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
              improved_plus_filler, improved_plus_question2, gap_improved,
              case_improved and factored (mode C only)
tune_threshold.py tune --model M --gold G --ids I [--masks K] [--grid 0.05:0.95:0.05] [--split NAME]
    [--train TRAIN.jsonl --train-ids TRAIN.ids] --out experiments/runs/tau_<mode>.json
    (the tau file records the split and the sha256 of every input)
    (--train/--train-ids add label priors and search gamma: prior-adjusted decoding)
tune_threshold.py predict --model M --input JSONL --ids I --tau-file experiments/runs/tau_<mode>.json
    --out PRED --manifest-out MAN

evaluate_models.py score --gold JSONL --ids IDS --predictions PRED [--masks K] --out REPORT.json
    [--manifest MAN] [--bootstrap N] [--seed S]
evaluate_models.py compare --gold JSONL --ids IDS [--masks K] --out REPORT.json
    --predictions-a A --predictions-b B [--bootstrap N] [--seed S] [--class LABEL ...]
    (--class is repeatable: per-class F1 of A and B, B - A and, with --bootstrap, its paired CI)
evaluate_models.py errors --gold JSONL --ids IDS --predictions PRED [--masks K] --out REPORT.json
    [--block NAME] [--per-pair K] [--seed S]
evaluate_models.py baseline --gold JSONL --ids IDS --out PRED.jsonl

run_demo.py --model-path MODEL [--text "some lowercased text"]

train_hidden_event_lm.py train --mode {A,B,C} --input JSONL --ids IDS --output MODEL.json
    [--order 3] [--discount 0.75] [--min-count 1] [--model-name NAME]
train_hidden_event_lm.py predict --model MODEL.json --input JSONL [--ids IDS] --output PRED.jsonl
    [--manifest-out MAN.json]
factor_predictions.py --punct A_PRED.jsonl --case CASE_PRED.jsonl --gold MODE_C_GOLD.jsonl --out C_PRED.jsonl
    --manifest-out MAN.json [--model-name factored]
simulate_asr_noise.py [--data-root data] [--release sponspeech-m1-v2] --split {validation,test} --wer W
    [--seed 2026] [--edit-model {v1,v2}] [--masks FILE] --out NEW_DIR

find_gold_noise.py [--data-root data] [--release sponspeech-m1-v2] [--mode B] --out NEW_DIR
reporting/asr_noise_table.py [--scores-root results/scores] --json-out OUT.json --out OUT.md
reporting/crf_convergence.py --models MODEL.joblib|DIR [...] --out OUT.json
reporting/length_position.py --gold MODE_A_GOLD.jsonl --ids IDS [--masks K]
    --predictions NAME=PRED.jsonl [NAME=PRED.jsonl ...] --out OUT.json

project_joint.py [--from {C,B}] --predictions PRED.jsonl --gold MODE_A_GOLD.jsonl --out A_PRED.jsonl
    --manifest-out A_PRED.manifest.json
build_leakage_subsets.py [--data-root data] [--release sponspeech-m1-v2] [--mode {A,B,C}]
    [--split {test,validation}] --out DIR

reporting/build_m3_tables.py [--scores-root results/scores] [--split {validation,test}] [--tau] --out OUT.md
reporting/build_m3_final_sections.py [--scores-root results/scores] [--manifests DIR] --out OUT.md
reporting/build_ablation_table.py [--scores-root results/scores] --out OUT.md [--setup TEXT]
reporting/time_models.py --kind {hmm,crf} --model MODEL --mode {A,B,C} --out OUT.json
    [--data-root data] [--release sponspeech-m1-v2]
    [--fit-train JSONL --fit-ids IDS --smoothing S --min-token-count N]
```

`--masks` is required in mode C (`masks/case_joint/<split>.jsonl`). `run_demo.py` exits 1
with one friendly line when the model file or the text is unusable. The data, HMM and
evaluation scripts refuse to overwrite existing outputs; `train_crf.py` replaces a file only
with `--overwrite`.
