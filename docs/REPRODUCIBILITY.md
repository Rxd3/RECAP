# Reproducibility guide

Every command below comes from the owning member's runbook: [data/README.md](../data/README.md),
[HMM_RUNBOOK.md](../experiments/HMM_RUNBOOK.md), [CRF_EXPERIMENTS.md](CRF_EXPERIMENTS.md) and
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md), plus the final-push CLIs
(`scripts/run_ablations.py`, `scripts/tune_threshold.py`, `scripts/run_demo.py`). Run everything
from the repository root. Each script prints its full options with `--help`; see also
[scripts/README.md](../scripts/README.md).

## Environments actually used

| Stage | Python | Key packages | Source |
| --- | --- | --- | --- |
| Data release `sponspeech-m1-v2` | CPython 3.13.5, Windows | PyYAML 6.0.3 | [DATA_RELEASE_VALIDATION.md](DATA_RELEASE_VALIDATION.md), [DECISIONS.md](DECISIONS.md) |
| HMM study and verification | CPython 3.13.5 | PyYAML 6.0.3 (`requirements-hmm.txt`) | [HMM_RESULTS.md](../experiments/HMM_RESULTS.md), [HMM_RUNBOOK.md](../experiments/HMM_RUNBOOK.md) |
| CRF full-data runs (Member 3) | 3.15.2 as recorded by Member 3's manifests; Member 5 used 3.13.1 to reproduce and predict | python-crfsuite 0.9.12, sklearn-crfsuite 0.5.0, joblib 1.6.0 | [CRF_NOTES.md](../experiments/CRF_NOTES.md) §1, [CRF_EXPERIMENTS.md](CRF_EXPERIMENTS.md) §2 |
| Scoring and runtime measurement (Member 5) | 3.13.1 | evaluator at the commits named in each report | [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) §7, [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §10 |

No GPU or pretrained model is needed. `python -m pip install -e ".[dev]"` installs the package,
its pinned runtime dependencies and pytest. `requirements-lock.txt` is the exact `pip freeze` of the
tested environment (CPython 3.13.1), including scikit-learn, SciPy and NumPy, through which the pickled
CRF models load; `python -m pip install -r requirements-lock.txt` recreates it.

**Hardware.** Training the improved CRF peaks at about 14.5 GB of RAM and takes 859–1,388 s per mode
on Member 3's machine; the basic CRF needs about 7 GB and 335–875 s. Prediction runs on a normal laptop: under 0.67 ms per
utterance for the improved CRF ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), Provenance and
§10). An HMM fit takes about 9 s ([HMM_RESULTS.md](../experiments/HMM_RESULTS.md)).

## Where the data and trained models are

| Item | Location | Check |
| --- | --- | --- |
| Data release `sponspeech-m1-v2` | team Drive [`dataset/`](https://drive.google.com/drive/folders/1jVl5pOQ6PvWPQ6MQxXLcbUksJJ1KDo5u), `sponspeech-m1-v2-handoff.zip` | SHA-256 `1e4b4a9504c68cd8…` ([data/README.md](../data/README.md)) |
| HMM bundle (three selected models and the full study) | team Drive [`Models/`](https://drive.google.com/drive/folders/13bjpo8GLVOwTJFJoOsG97O_wSyVF5rsN), `hmm-sponspeech-v2-20261029.zip` | SHA-256 `4cdc5725b6538d87…`, 27,095,563 bytes ([HMM_RESULTS.md](../experiments/HMM_RESULTS.md)) |
| CRF models (six `.joblib` files) | [`RECAP_M3_DRIVE_MODELS`](https://drive.google.com/drive/folders/1jSfe5U8yZJxseU7aA9p6ycQNoZCtx7e_), as shared by Member 3, with a `SHA256SUMS` file | per-model SHA-256 in [CRF_EXPERIMENTS.md](CRF_EXPERIMENTS.md) §7 |
| Evaluator outputs behind every number | committed in `results/scores/` | score, compare and errors JSON record their input SHA-256s; τ files record the model's SHA-256 and runtime files the model file name ([results/scores/README.md](../results/scores/README.md)) |

Code blocks marked `bash` use `\` to continue a line (bash or Git Bash on Windows); blocks marked
`powershell` use a backtick.

## 1. Environment and tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

On macOS/Linux, activate with `source .venv/bin/activate`. The tests use small fixtures and
download nothing. Checked on 2026-10-30: a fresh `git clone` of the final `release/final` branch passes
`python -m pytest -q` (1302 tests) with the pinned dependency versions installed. The lock file was
produced on Python 3.13.1.

## 2. Data

Rebuild release `sponspeech-m1-v2` from the official archive. First unpack the provenance
package (raw manifest and title snapshot) into the raw root, as [data/README.md](../data/README.md)
describes.

```bash
python scripts/prepare_data.py extract --raw-root data/raw/sponspeech-gcs-generation-1716975130398953
python scripts/prepare_data.py build --raw-root data/raw/sponspeech-gcs-generation-1716975130398953 \
  --release-id sponspeech-m1-v2
python -m pytest
```

Then compare `data/processed/<release>/SHA256SUMS` and `data/splits/<release>/SHA256SUMS` with
the published copies. If the byte ranges can no longer be read, add `--archive <path>/sponspeech.tar.gz`
to `extract` (the full 63.5 GB archive). To use the team's archive instead of rebuilding:

```bash
shasum -a 256 -c <id>.SHA256SUMS
tar -xzf <id>-processed.tar.gz -C data/processed
tar -xzf <id>-splits.tar.gz -C data/splits
(cd data/processed/<id> && shasum -a 256 -c SHA256SUMS)
(cd data/splits/<id> && shasum -a 256 -c SHA256SUMS)
```

The expected fingerprints are in [DATA_RELEASE_VALIDATION.md](DATA_RELEASE_VALIDATION.md).

## 3. Shortcut baseline

```bash
python scripts/evaluate_models.py baseline --gold <processed>/mode_A/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_A.ids --out results/predictions/shortcut_A_validation.jsonl
```

`<processed>` is `data/processed/sponspeech-m1-v2` and `<splits>` is `data/splits/sponspeech-m1-v2`.

## 4. HMM

The HMM study uses Member 5's evaluator pinned in a separate checkout:

```powershell
python -m venv tmp/hmm-env
tmp/hmm-env/Scripts/python.exe -m pip install -r requirements-hmm.txt
git fetch origin feat/m5-evaluation
git worktree add --detach tmp/member2-evaluation-3044a58 3044a58e777200bcac64e6e3364922404be359bf
```

Tune on validation, finalize on test, verify, then package:

```powershell
tmp/hmm-env/Scripts/python.exe scripts/run_hmm_experiments.py tune --config configs/hmm_sponspeech_v2.yaml --output-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58
tmp/hmm-env/Scripts/python.exe scripts/run_hmm_experiments.py finalize --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58
tmp/hmm-env/Scripts/python.exe scripts/verify_hmm_experiments.py --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58 --output-dir experiments/runs/hmm-sponspeech-v2-20261029/verification
tmp/hmm-env/Scripts/python.exe scripts/package_hmm_handoff.py --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58 --output-dir artifacts/hmm/hmm-sponspeech-v2-20261029
```

Output directories must be new; the completed study is never overwritten. On Linux/macOS use
`tmp/hmm-env/bin/python`.

## 5. CRF

Train one model (repeat for each mode and preset; the six runs and their hashes are in
[CRF_EXPERIMENTS.md](CRF_EXPERIMENTS.md) §7):

```powershell
.\.venv\Scripts\python.exe scripts/train_crf.py train `
  --mode <A|B|C> `
  --input data/processed/sponspeech-m1-v2/mode_<A|B|C>/train.jsonl `
  --ids data/splits/sponspeech-m1-v2/default_train/mode_<A|B|C>.ids `
  --output experiments/runs/<run_dir>/<run_id>.joblib `
  --manifest-out experiments/runs/<run_dir>/<run_id>.manifest.json `
  --preset <basic|improved> `
  --algorithm lbfgs `
  --c1 0.1 --c2 0.1 --max-iterations 100 `
  --run-id <run_id> `
  --dataset-release sponspeech-m1-v2 `
  --policy-version m1-v2
```

Predict the primary validation list with the prediction manifest:

```powershell
.\.venv\Scripts\python.exe scripts/train_crf.py predict `
  --model experiments/runs/<run_dir>/<run_id>.joblib `
  --input data/processed/sponspeech-m1-v2/mode_<A|B|C>/validation.jsonl `
  --ids data/splits/sponspeech-m1-v2/primary_eval/validation_mode_<A|B|C>.ids `
  --output experiments/runs/<run_dir>/<run_id>_val_predictions.jsonl `
  --manifest-out experiments/runs/<run_dir>/<run_id>_val_predictions.manifest.json
```

Test and test_amb use the same command, run once per frozen model: `--input .../test.jsonl`
with `--ids .../primary_eval/test_mode_<X>.ids`, and `--input .../test_amb.jsonl` with
`--ids .../primary_eval/test_amb_mode_<X>.ids`.

## 6. Scoring

```bash
python scripts/evaluate_models.py score --gold <processed>/mode_C/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_C.ids --predictions <model predictions>.jsonl \
    --masks <processed>/masks/case_joint/validation.jsonl --out results/tables/<run_id>.json
```

Modes A and B take the same command without `--masks`. Add `--bootstrap 1000` for the 95%
intervals (seed 2026 by default) and `--manifest <predictions>.manifest.json` to check the
prediction manifest. Paired comparisons and error examples:

```bash
python scripts/evaluate_models.py compare --gold <gold> --ids <ids> --out <report>.json \
    --predictions-a <A>.jsonl --predictions-b <B>.jsonl --bootstrap 1000
python scripts/evaluate_models.py errors --gold <gold> --ids <ids> --predictions <pred>.jsonl \
    --out <report>.json --per-pair 5
```

Per-class paired intervals (hypotheses H2–H4) add one `--class` per label to `compare`:

```bash
python scripts/evaluate_models.py compare --gold <processed>/mode_A/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_A.ids --predictions-a <A>.jsonl --predictions-b <B>.jsonl \
    --bootstrap 1000 --seed 2026 --class COMMA --class QUESTION --out experiments/runs/per_class.json
```

`score` never overwrites an output and never creates folders.

## 7. Ablations and threshold decoding (validation only)

```bash
python scripts/run_ablations.py --mode A --out experiments/runs/ablations_A --sample 50000 \
    --data-root data --release sponspeech-m1-v2
python scripts/tune_threshold.py tune --model <model>.joblib --gold <processed>/mode_A/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_A.ids --grid 0.05:0.95:0.05 --out experiments/runs/tau_A.json
python scripts/tune_threshold.py predict --model <model>.joblib --input <processed>/mode_A/test.jsonl \
    --ids <splits>/primary_eval/test_mode_A.ids --tau-file experiments/runs/tau_A.json \
    --out experiments/runs/crf_improved_tau_A_test.jsonl \
    --manifest-out experiments/runs/crf_improved_tau_A_test.manifest.json
```

Write every output under the ignored `experiments/runs/` (or `results/`): an untracked file at the
repository root makes a later HMM `tune` refuse to start, because it requires a clean working tree.
Modes B and C use `tau_B.json` and `tau_C.json` in the same folder.

`run_ablations.py` takes `--variants` to run a subset (`improved`,
`improved_minus_{window2,ngrams,question,clause,casing,position}`, `improved_plus_filler`).
`--data-root` (default `data`) and `--release` (default `sponspeech-m1-v2`) locate the release.
It writes `DIR/mode_X/sample.ids` and one `DIR/mode_X/<variant>/validation.jsonl` per variant and
never reads test; score each file with `evaluate_models.py score` (section 6).
Mode C `tune` also takes `--masks <processed>/masks/case_joint/validation.jsonl`. τ is chosen on
validation; test is predicted once with the frozen τ.

**Sensitivity of the fixed CRF settings** ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §5):
retrain both presets on the 50k sample that `run_ablations.py` wrote, at each setting, then predict and
score validation only:

```bash
python scripts/train_crf.py train --mode A --input <processed>/mode_A/train.jsonl \
    --ids experiments/runs/ablations_A/mode_A/sample.ids --preset improved \
    --c1 1.0 --c2 1.0 --max-iterations 300 --output experiments/runs/sensitivity/improved_c1.0_i300.joblib
python scripts/train_crf.py predict --model experiments/runs/sensitivity/improved_c1.0_i300.joblib \
    --input <processed>/mode_A/validation.jsonl --ids <splits>/primary_eval/validation_mode_A.ids \
    --output experiments/runs/sensitivity/improved_c1.0_i300.jsonl
```

Repeat with `--preset basic` and c1 = c2 ∈ {0.01, 0.1, 1.0}, 100 or 300 iterations, then compare each
improved model with its basic twin (`evaluate_models.py compare --bootstrap 1000`).

## 8. Studies: joint vs punctuation-only (H1), leakage subsets and label granularity

```bash
python scripts/project_joint.py --predictions <mode C validation predictions>.jsonl \
    --gold <processed>/mode_A/validation.jsonl \
    --out experiments/runs/h1_A_from_C.jsonl --manifest-out experiments/runs/h1_A_from_C.manifest.json
python scripts/evaluate_models.py compare --gold <processed>/mode_A/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_C.ids \
    --predictions-a <mode A validation predictions>.jsonl --predictions-b experiments/runs/h1_A_from_C.jsonl \
    --bootstrap 1000 --seed 2026 --out experiments/runs/h1_compare.json
python scripts/build_leakage_subsets.py --mode A --split test --out experiments/runs/leakage_test_A
```

The leakage command writes `all.ids`, `kept.ids`, `excluded_train_video.ids`, `excluded_other_leak.ids`
and `excluded_quality.ids` with `counts.json`. For mode A test the counts are 17,695 / 12,668 / 2,706 /
2,130 / 191 ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §8). Score each list with
`evaluate_models.py score --ids`. The "all excluded" set (5,027 utterances for mode A test) is the
union of the three `excluded_*.ids` lists, for example
`cat excluded_train_video.ids excluded_other_leak.ids excluded_quality.ids > excluded_all.ids`
(the three lists are disjoint: 2,706 + 2,130 + 191 = 5,027).

**Label granularity, B→A** ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §11): project each mode B
model's validation predictions to mode A with the D5 mapping, then compare with its mode A twin:

```bash
python scripts/project_joint.py --from B --predictions <mode B validation predictions>.jsonl \
    --gold <processed>/mode_A/validation.jsonl \
    --out experiments/runs/b_to_a.jsonl --manifest-out experiments/runs/b_to_a.manifest.json
python scripts/evaluate_models.py compare --gold <processed>/mode_A/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_A.ids \
    --predictions-a <mode A validation predictions>.jsonl --predictions-b experiments/runs/b_to_a.jsonl \
    --bootstrap 1000 --seed 2026 --class COMMA --class PERIOD --class QUESTION --out experiments/runs/b_to_a_compare.json
```

## 9. Tables and runtime

```bash
python scripts/reporting/build_m3_tables.py --split test --tau --out experiments/runs/crf_test_tables.md
python scripts/reporting/build_m3_final_sections.py --out experiments/runs/crf_sections_7_10.md
python scripts/reporting/build_ablation_table.py --out experiments/runs/ablation_tables.md
python scripts/reporting/time_models.py --kind crf --model <model>.joblib --mode A --out experiments/runs/runtime_crf_A.json
```

The first three read only the committed JSON in `results/scores/` and the committed training manifests
(`--manifests` defaults to `results/scores/crf_training_manifests`);
they reproduce the tables of [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) (§1–4 with errors and
confusion matrices, §7–10, §6). `time_models.py` needs a saved model and the release.

## 10. Demo

The demo takes any saved model. To fit a small HMM `.json` for it (full mode C training list, about
9 s, with the selected mode C settings), then restore one sentence:

```bash
python scripts/train_hmm.py --train data/processed/sponspeech-m1-v2/mode_C/train.jsonl \
    --train-ids data/splits/sponspeech-m1-v2/default_train/mode_C.ids \
    --output-dir experiments/runs/hmm_demo_C --label-mode C --smoothing 0.01 --min-token-count 2 \
    --dataset-release-id sponspeech-m1-v2 --preprocessing-version m1-v2 --tokenizer-version m1-v2 \
    --label-policy-version m1-v2 --split-manifest data/splits/sponspeech-m1-v2/SHA256SUMS \
    --release-manifest data/processed/sponspeech-m1-v2/release_manifest.json
python scripts/run_demo.py --model-path experiments/runs/hmm_demo_C/model.json --text "hello everyone welcome to our presentation"
```

Mode C restores casing too; a mode A model (`mode_A` files, `--label-mode A --min-token-count 1`)
restores punctuation only. A CRF `.joblib` from the Drive folder above works the same way.

Without `--text`, the demo prompts for input. A missing or unsupported model, or unusable text,
exits 1 with one `error:` line.

## Reproduction levels reached

1. **Recompute metrics from saved predictions.** Every reported table can be recomputed or checked
   from the evaluator JSON committed in `results/scores/` ([results/scores/README.md](../results/scores/README.md)):
   the `scripts/reporting/` builders regenerate the CRF tables from those files alone (section 9).
   HMM scores equal those in its own bundle ([EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)). For the CRF, an independent from-scratch scorer
   confirmed the validation numbers only ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), Provenance).
2. **Rerun inference from saved models.** HMM: every full-split prediction was reproduced
   ([HMM_RUNBOOK.md](../experiments/HMM_RUNBOOK.md)). CRF: all six validation prediction files were
   regenerated byte for byte from the shared, sha256-verified `.joblib` models
   ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), Provenance).
3. **Retrain.** HMM: each selected model was refit and matched byte for byte. CRF: mode A basic was
   retrained from a clean `aa232be` checkout, and its predictions are byte-identical to the
   delivered file ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), Provenance).

A saved model does not prove its own training provenance; never update a checksum to hide a
mismatch.

## 11. Neural comparisons

These optional comparisons leave the classical models and evaluator unchanged. See
[NEURAL_NOTES.md](../experiments/NEURAL_NOTES.md) for current delivery status, alignment,
selection and measurement details. No neural result is claimed until its run completes.
Use a separate environment; existing dependencies retain their pins:

```powershell
python -m venv .venv-neural
.venv-neural/Scripts/Activate.ps1
# CUDA 12.8 build supports the RTX 5070; verify the GPU before starting.
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev,neural]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q
```

For CPU-only, download-free verification, omit the CUDA installation and install
`.[dev,neural]` from PyPI. Also run the full suite in a fresh `.[dev]` environment
without torch: optional tensor tests skip, while alignment/protocol tests still run.

Download and verify `sponspeech-m1-v2-handoff.zip` from the `dataset/` link above,
then unpack it as [data/README.md](../data/README.md#getting-the-release-from-google-drive)
describes. The following loop reads only training and validation. It searches both
BERT LRs (2e-5, 5e-5), and epochs 1–3 for both families, using T5 LR 1e-4:

```powershell
$neuralProcessed = "data/processed/sponspeech-m1-v2"
$neuralSplits = "data/splits/sponspeech-m1-v2"
foreach ($neuralMode in @("A", "B", "C")) {
  foreach ($neuralFamily in @("bert_tagger", "t5_seq2seq")) {
    $neuralScript = if ($neuralFamily -eq "bert_tagger") { "scripts/train_neural_tagger.py" } else { "scripts/train_seq2seq.py" }
    $neuralRun = "experiments/runs/neural/${neuralFamily}_${neuralMode}"
    $neuralMaskArgs = if ($neuralMode -eq "C") { @("--masks", "$neuralProcessed/masks/case_joint/validation.jsonl") } else { @() }
    python $neuralScript train --mode $neuralMode `
      --input "$neuralProcessed/mode_$neuralMode/train.jsonl" `
      --ids "$neuralSplits/default_train/mode_$neuralMode.ids" `
      --validation-input "$neuralProcessed/mode_$neuralMode/validation.jsonl" `
      --validation-ids "$neuralSplits/primary_eval/validation_mode_$neuralMode.ids" `
      --output $neuralRun --manifest-out "$neuralRun/training.manifest.json" `
      --selection-out "$neuralRun/selection.json" --run-id "${neuralFamily}_${neuralMode}" `
      --epochs 3 --batch-size 4 --gradient-accumulation 4 @neuralMaskArgs
    if ($LASTEXITCODE -ne 0) { throw "Neural training failed; inspect the retained run evidence" }
  }
}
```

All run directories must be new. Set `--pretrained t5-small` on the seq2seq command
only if memory is short, and record that fallback in a fresh run. Batch size,
accumulation, LR (T5: 1e-4 or 3e-4) and beams (1 or 4) must be fixed using validation.
Never change them using test feedback. BERT uses fp16 on CUDA; T5 uses bf16 when
available and otherwise float32. No automatic memory fallback changes the experiment.

Each completed run writes candidate artifacts/predictions/scores, the frozen
`selection.json`, final `validation.jsonl` and `validation.manifest.json`, a
bootstrap-1000 validation score, and a training manifest including the median of
three validation timing passes after warm-up. It appends the full selection and
runtime record to `experiments/NEURAL_NOTES.md`; commit those real records before
opening held-out splits. The notes currently state which work remains pending.

After all six selections are frozen and recorded, predict each held-out split once:

```powershell
foreach ($neuralMode in @("A", "B", "C")) {
  foreach ($neuralFamily in @("bert_tagger", "t5_seq2seq")) {
    $neuralScript = if ($neuralFamily -eq "bert_tagger") { "scripts/train_neural_tagger.py" } else { "scripts/train_seq2seq.py" }
    $neuralRun = "experiments/runs/neural/${neuralFamily}_${neuralMode}"
    $neuralSelection = Get-Content "$neuralRun/selection.json" -Raw | ConvertFrom-Json
    $neuralModel = "$neuralRun/$($neuralSelection.selected.artifact)"
    foreach ($neuralSplit in @("test", "test_amb")) {
      python $neuralScript predict --model $neuralModel --selection "$neuralRun/selection.json" `
        --input "$neuralProcessed/mode_$neuralMode/$neuralSplit.jsonl" `
        --ids "$neuralSplits/primary_eval/${neuralSplit}_mode_$neuralMode.ids" `
        --output "$neuralRun/$neuralSplit.jsonl" `
        --manifest-out "$neuralRun/$neuralSplit.manifest.json" --batch-size 4
      if ($LASTEXITCODE -ne 0) { throw "Neural prediction failed; inspect the receipt before continuing" }
    }
  }
}
python scripts/package_neural_handoff.py --run-root experiments/runs/neural `
  --output artifacts/neural/neural-predictions.zip
```

Held-out inference writes exclusive attempt receipts to prevent accidental repeats.
The package command validates all six freeze records, scores all 18 pairs using
`evaluate_models.py score --manifest ... --bootstrap 1000` (with split-specific C
masks), and writes the 36 prediction/manifest files plus `SHA256SUMS` to the ZIP.
It keeps score JSON/Markdown beside the ZIP; all are ignored by Git. Upload the ZIP
to team Drive `Models/` and send its link to Abd Raouf (Member 5). Data, weights
and predictions must never be committed.
