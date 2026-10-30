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
| CRF full-data runs (Member 3) | 3.15.2, as recorded in the delivered training manifests | python-crfsuite 0.9.12, sklearn-crfsuite 0.5.0, joblib 1.6.0 | [CRF_NOTES.md](../experiments/CRF_NOTES.md) §1, [CRF_EXPERIMENTS.md](CRF_EXPERIMENTS.md) §2 |
| Scoring and runtime measurement (Member 5) | 3.13.1 | evaluator at the commits named in each report | [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) §7, [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §10 |

No GPU or pretrained model is needed. `python -m pip install -e ".[dev]"` installs the package,
its pinned runtime dependencies and pytest.

## 1. Environment and tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

On macOS/Linux, activate with `source .venv/bin/activate`. The tests use small fixtures and
download nothing.

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

```text
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
  --ids data/splits/sponspeech-m1-v2/default_train/mode_<A|C>.ids `
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
  --ids data/splits/sponspeech-m1-v2/primary_eval/validation_mode_<A|C>.ids `
  --output experiments/runs/<run_dir>/<run_id>_val_predictions.jsonl `
  --manifest-out experiments/runs/<run_dir>/<run_id>_val_predictions.manifest.json
```

Test and test_amb use the same command, run once per frozen model: `--input .../test.jsonl`
with `--ids .../primary_eval/test_mode_<X>.ids`, and `--input .../test_amb.jsonl` with
`--ids .../primary_eval/test_amb_mode_<X>.ids`.

## 6. Scoring

```text
python scripts/evaluate_models.py score --gold <processed>/mode_C/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_C.ids --predictions <model predictions>.jsonl \
    --masks <processed>/masks/case_joint/validation.jsonl --out results/tables/<run_id>.json
```

Modes A and B take the same command without `--masks`. Add `--bootstrap 1000` for the 95%
intervals (seed 2026 by default) and `--manifest <predictions>.manifest.json` to check the
prediction manifest. Paired comparisons and error examples:

```text
python scripts/evaluate_models.py compare --gold <gold> --ids <ids> --out <report>.json \
    --predictions-a <A>.jsonl --predictions-b <B>.jsonl --bootstrap 1000
python scripts/evaluate_models.py errors --gold <gold> --ids <ids> --predictions <pred>.jsonl \
    --out <report>.json --per-pair 5
```

`score` never overwrites an output and never creates folders.

## 7. Ablations and threshold decoding (validation only)

```text
python scripts/run_ablations.py --mode A --out experiments/runs/ablations_A --sample 50000 \
    --data-root data --release sponspeech-m1-v2
python scripts/tune_threshold.py tune --model <model>.joblib --gold <processed>/mode_A/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_A.ids --grid 0.05:0.95:0.05 --out tau_A.json
python scripts/tune_threshold.py predict --model <model>.joblib --input <processed>/mode_A/test.jsonl \
    --ids <splits>/primary_eval/test_mode_A.ids --tau-file tau_A.json \
    --out <predictions>.jsonl --manifest-out <predictions>.manifest.json
```

`run_ablations.py` takes `--variants` to run a subset (`improved`,
`improved_minus_{window2,ngrams,question,clause,casing,position}`, `improved_plus_filler`).
`--data-root` (default `data`) and `--release` (default `sponspeech-m1-v2`) locate the release.
It writes `DIR/mode_X/sample.ids` and one `DIR/mode_X/<variant>/validation.jsonl` per variant and
never reads test; score each file with `evaluate_models.py score` (section 6).
Mode C `tune` also takes `--masks <processed>/masks/case_joint/validation.jsonl`. τ is chosen on
validation; test is predicted once with the frozen τ.

## 8. Demo

```text
python scripts/run_demo.py --model-path <model.json or model.joblib> --text "hello everyone welcome to our presentation"
```

Without `--text`, the demo prompts for input. A missing or unsupported model, or unusable text,
exits 1 with one `error:` line.

## Reproduction levels reached

1. **Recompute metrics from saved predictions.** HMM scores equal those in its own bundle
   ([EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)). For the CRF, an independent from-scratch scorer
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
