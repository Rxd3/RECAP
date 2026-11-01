# Scripts (compatibility stubs)

The commands live in the package and run as `recap <command>` after `python -m pip install -e .`;
`recap --help` lists them. Each file here is a 3-line stub that imports the command's `main` from
`punct_restore.cli` (or `punct_restore.reporting`) and runs it, so an old command line such as
`scripts/train_crf.py train ...` run with Python still works and behaves exactly like
`recap train-crf train ...`. The command name is the module name with `_` written as `-`. Full command
sequences are in [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).

The command modules only parse arguments, read and write files, and report errors as one line on
stderr; the logic lives in the library modules of `src/punct_restore/`.

| Command | Stub | Owner | Stage | What it does |
| --- | --- | --- | --- | --- |
| `recap prepare-data` | `prepare_data.py` | Member 1 | Data | `extract` recreates the SponSpeech transcripts from pinned byte ranges of the official archive (`--archive` for the full-archive fallback); `build` writes a validated, immutable release with its ID lists. |
| `recap train-hmm` | `train_hmm.py` | Member 2 | HMM | Trains the HMM on an explicit `--train-ids` list; writes a JSON artifact and run manifest, and optional predictions (`--predict-input`, `--predict-ids`). `--release-manifest` checks the inputs against the release. |
| `recap predict-hmm` | `predict_hmm.py` | Member 2 | HMM | Predicts with a saved artifact (`--ids` or `--all-records`), without training. |
| `recap run-hmm-experiments` | `run_hmm_experiments.py` | Member 2 | HMM | `tune`: frozen validation search from `configs/hmm_sponspeech_v2.yaml`. `finalize`: predicts and scores the selected models on validation, test and test_amb. |
| `recap verify-hmm-experiments` | `verify_hmm_experiments.py` | Member 2 | HMM | Independent audit of a study: score recounts, refits of the winners, prediction comparison. |
| `recap package-hmm-handoff` | `package_hmm_handoff.py` | Member 2 | HMM | Packages a verified study as a ZIP with internal and outer SHA-256 sums. |
| `recap train-crf` | `train_crf.py` | Member 3 | CRF | `train` fits a linear-chain CRF (`--preset basic` or `improved`) on an explicit `--ids` list and writes a `.joblib` model and manifest. `predict` writes prediction JSONL and, with `--manifest-out`, the prediction manifest. `inspect` prints an artifact's metadata. |
| `recap run-ablations` | `run_ablations.py` | Team (final push) | CRF | Trains the improved CRF with one feature group removed (or the filler cue added) on a training sample and writes each variant's primary-validation predictions; score them with `recap evaluate-models`. `improved_plus_clusters` adds the word-cluster features and needs `--cluster-map` (a map from `recap build-word-clusters`; its mode, release and train-ids sha256 must match). Never reads test. |
| `recap build-word-clusters` | `build_word_clusters.py` | Team (final push) | CRF (features) | Builds a word-cluster map from `default_train` only (lowercased words with count ≥ `--min-count`, ±`--window` co-occurrences, PPMI, TruncatedSVD to `--dims`, k-means with `--k` and `--seed`) and writes it as JSON with the train-ids sha256, k, the seed and the vocabulary size. `--out` must not exist. |
| `recap tune-threshold` | `tune_threshold.py` | Team (final push) | CRF | Threshold decoding of CRF marginals: `tune` picks τ on validation, `predict` applies the frozen τ once. |
| `recap evaluate-models` | `evaluate_models.py` | Member 5 | Evaluation | `score` one prediction file, `compare` two on the same IDs (paired bootstrap), `errors` (confusion pairs and examples), `baseline` (the end-of-utterance shortcut). |
| `recap project-joint` | `project_joint.py` | Team (final push) | Study (H1, B→A) | Projects mode C (`--from C`, default; `model_name` `<name>_joint_C`) or mode B predictions (`--from B`, the approved D5 mapping; `<name>_from_B`) onto mode A punctuation labels, with a manifest. |
| `recap build-leakage-subsets` | `build_leakage_subsets.py` | Team (final push) | Study (leakage) | Writes `all.ids`, `kept.ids`, `excluded_train_video.ids`, `excluded_other_leak.ids`, `excluded_quality.ids` and `counts.json` for one split and mode. |
| `recap build-m3-tables` | `reporting/build_m3_tables.py` | Team (final push) | Reporting | CRF result tables for one split from `results/scores/`; `--tau` adds the threshold rows (test only). |
| `recap build-m3-final-sections` | `reporting/build_m3_final_sections.py` | Team (final push) | Reporting | Threshold, leakage, test_amb and runtime sections from `results/scores/` and the training manifests (`--manifests`, default `<scores-root>/crf_training_manifests`). |
| `recap build-ablation-table` | `reporting/build_ablation_table.py` | Team (final push) | Reporting | Feature-ablation tables from `results/scores/ablations/`. |
| `recap time-models` | `reporting/time_models.py` | Team (final push) | Reporting | Median-of-3 prediction timing after a 200-record warm-up (and an optional HMM fit). |
| `recap train-hidden-event-lm` | `train_hidden_event_lm.py` | Team (final push) | Comparison | `train` fits a Stolcke-style hidden-event LM (mode C adds a unigram truecaser); `predict` writes predictions and, with `--manifest-out`, a manifest. |
| `recap factor-predictions` | `factor_predictions.py` | Team (final push) | Comparison | Joins mode A predictions and CASE predictions (a casing-only CRF) into mode C predictions with a manifest. |
| `recap simulate-asr-noise` | `simulate_asr_noise.py` | Team (final push) | Robustness | Copies the validation or test inputs of every mode with simulated ASR word errors (`--edit-model {v1,v2}`: `v1` has deletion, homophone substitution, adjacent swap; `v2` adds `default_train` confusion substitution, filler/function-word insertion, <=3-char deletion, and 10% swap cap) at a target WER; labels and casing masks stay aligned. |
| `recap find-gold-noise` | `find_gold_noise.py` | Team (final push) | Study (gold noise) | Flags speaker labels, subtitle cue residue and auto-caption videos on the primary mode B lists; writes `<split>_B_noisy.tsv` and `<split>_B_summary.json`. |
| `recap asr-noise-table` | `reporting/asr_noise_table.py` | Team (final push) | Reporting | Builds the multi-seed ASR noise `v2` summary JSON and `docs/ASR_NOISE_V2.md` from `results/scores/`. |
| `recap crf-convergence` | `reporting/crf_convergence.py` | Team (final push) | Reporting | Loss-trace summary of trusted CRF `.joblib` models. |
| `recap length-position` | `reporting/length_position.py` | Team (final push) | Reporting | Mode A micro F1 by utterance length and QUESTION recall by position (EVALUATION_RESULTS.md §6). |
| `recap run-demo` | `run_demo.py` | Member 6 | Demo | Loads a saved HMM or hidden-event LM (`.json`), or a CRF or gap classifier (`.joblib`, `.pkl`), through `src/punct_restore/pipeline.py` (CASE-mode CRFs are refused) and prints the restored text; without `--text` it prompts interactively. |

## Interfaces

Flags below are the ones each command accepts (checked with `recap <command> --help`; see it for
defaults).

```text
recap prepare-data extract --raw-root RAW [--archive sponspeech.tar.gz]
recap prepare-data build   --raw-root RAW --release-id ID [--processed-dir DIR] [--splits-dir DIR]

recap train-hmm --train JSONL --train-ids IDS --output-dir DIR [--label-mode {A,B,C}]
    [--smoothing S] [--min-token-count N] --dataset-release-id ID --preprocessing-version V
    --tokenizer-version V --label-policy-version V --split-manifest PATH
    [--predict-input JSONL] [--predict-ids IDS] [--release-manifest PATH]
recap predict-hmm --model PATH --input JSONL --output-dir DIR (--ids IDS | --all-records)
    [--release-manifest PATH] [--selection-record PATH]
recap run-hmm-experiments tune --config YAML --output-dir DIR --evaluator-root DIR
recap run-hmm-experiments finalize --run-dir DIR --evaluator-root DIR
recap verify-hmm-experiments --run-dir DIR --evaluator-root DIR --output-dir DIR
recap package-hmm-handoff --run-dir DIR --evaluator-root DIR --output-dir DIR

recap train-crf train --mode {A,B,C} --input JSONL --ids IDS --output MODEL.joblib
    [--preset {basic,improved}] [--c1 C1] [--c2 C2] [--max-iterations N]
    [--run-id ID] [--manifest-out PATH] [--overwrite]
recap train-crf predict --model MODEL.joblib --input JSONL --output PRED.jsonl [--ids IDS]
    [--manifest-out PATH] [--overwrite]
recap train-crf inspect --model MODEL.joblib
recap run-ablations --mode {A,B,C} --out DIR [--sample 50000] [--variants V1,V2,...]
    [--cluster-map MAP.json] [--data-root data] [--release sponspeech-m1-v2]
    variants: improved, improved_minus_{window2,ngrams,question,clause,casing,position},
              improved_plus_filler, improved_plus_question2, improved_plus_clusters
              (needs --cluster-map), gap_improved, case_improved and factored (mode C only)
recap build-word-clusters [--data-root data] [--release sponspeech-m1-v2] [--mode {A,B,C}] --k K
    [--seed 2026] [--min-count 5] [--dims 100] [--window 2] --out MAP.json
recap tune-threshold tune --model M --gold G --ids I [--masks K] [--grid 0.05:0.95:0.05] [--split NAME]
    [--train TRAIN.jsonl --train-ids TRAIN.ids] --out experiments/runs/tau_<mode>.json
    (the tau file records the split and the sha256 of every input)
    (--train/--train-ids add label priors and search gamma: prior-adjusted decoding)
recap tune-threshold predict --model M --input JSONL --ids I --tau-file experiments/runs/tau_<mode>.json
    --out PRED --manifest-out MAN

recap evaluate-models score --gold JSONL --ids IDS --predictions PRED [--masks K] --out REPORT.json
    [--manifest MAN] [--bootstrap N] [--seed S]
recap evaluate-models compare --gold JSONL --ids IDS [--masks K] --out REPORT.json
    --predictions-a A --predictions-b B [--bootstrap N] [--seed S] [--class LABEL ...]
    (--class is repeatable: per-class F1 of A and B, B - A and, with --bootstrap, its paired CI)
recap evaluate-models errors --gold JSONL --ids IDS --predictions PRED [--masks K] --out REPORT.json
    [--block NAME] [--per-pair K] [--seed S]
recap evaluate-models baseline --gold JSONL --ids IDS --out PRED.jsonl

recap run-demo --model-path MODEL [--text "some lowercased text"]

recap train-hidden-event-lm train --mode {A,B,C} --input JSONL --ids IDS --output MODEL.json
    [--order 3] [--discount 0.75] [--min-count 1] [--model-name NAME]
recap train-hidden-event-lm predict --model MODEL.json --input JSONL [--ids IDS] --output PRED.jsonl
    [--manifest-out MAN.json]
recap factor-predictions --punct A_PRED.jsonl --case CASE_PRED.jsonl --gold MODE_C_GOLD.jsonl --out C_PRED.jsonl
    --manifest-out MAN.json [--model-name factored]
recap simulate-asr-noise [--data-root data] [--release sponspeech-m1-v2] --split {validation,test} --wer W
    [--seed 2026] [--edit-model {v1,v2}] [--masks FILE] --out NEW_DIR

recap find-gold-noise [--data-root data] [--release sponspeech-m1-v2] [--mode B] --out NEW_DIR
recap asr-noise-table [--scores-root results/scores] --json-out OUT.json --out OUT.md
recap crf-convergence --models MODEL.joblib|DIR [...] --out OUT.json
recap length-position --gold MODE_A_GOLD.jsonl --ids IDS [--masks K]
    --predictions NAME=PRED.jsonl [NAME=PRED.jsonl ...] --out OUT.json

recap project-joint [--from {C,B}] --predictions PRED.jsonl --gold MODE_A_GOLD.jsonl --out A_PRED.jsonl
    --manifest-out A_PRED.manifest.json
recap build-leakage-subsets [--data-root data] [--release sponspeech-m1-v2] [--mode {A,B,C}]
    [--split {test,validation}] --out DIR

recap build-m3-tables [--scores-root results/scores] [--split {validation,test}] [--tau] --out OUT.md
recap build-m3-final-sections [--scores-root results/scores] [--manifests DIR] --out OUT.md
recap build-ablation-table [--scores-root results/scores] --out OUT.md [--setup TEXT]
recap time-models --kind {hmm,crf} --model MODEL --mode {A,B,C} --out OUT.json
    [--data-root data] [--release sponspeech-m1-v2]
    [--fit-train JSONL --fit-ids IDS --smoothing S --min-token-count N]
```

`--masks` is required in mode C (`masks/case_joint/<split>.jsonl`). `recap run-demo` exits 1
with one friendly line when the model file or the text is unusable. The data, HMM and
evaluation commands refuse to overwrite existing outputs; `recap train-crf` replaces a file only
with `--overwrite`.
