# Project 1 reference review

Read-only inspection of [Rxd3/NALTRA](https://github.com/Rxd3/NALTRA/) on
2026-10-28; observed main commit
`bc0c40fdb2267445ffccc8060ec684ca0627b286`. No clone, modification or push to
NALTRA was performed. Sources were read through GitHub pages/raw files and APIs.

## Inspected material

- Repository tree and README: overview/task, quick start, models, results,
  reproduction and repository map; `src/naltra` with data/models/pipeline/evaluation.
- `pyproject.toml` and `requirements.txt`: setuptools `src/` packaging,
  supported dependencies versus recorded experiment pins, development extras.
- `CONTRIBUTING.md`, `docs/team_ownership.md`: focused branches, PR review,
  typed shared interfaces, primary reviewers, coordination of schema changes.
- `docs/reproducibility.md`, `scripts/README.md`, `results/README.md`: separate
  preparation/training/prediction/scoring scripts, versioned splits, manifests,
  provenance, and separation of model artifacts and large prediction dumps.
- Config/document directory names and presentation paths: YAML configuration,
  documentation guides and `docs/presentation/NALTRA_final.pptx`. The presentation
  binary itself was not opened or assessed.
- Branch list and five recent PR records: feature/fix branches merged into
  `main`, consistent with the documented workflow. Branch-protection settings
  and the full development history were not inspected.

## Reused and adapted

Reuse the `src/` Python package, snake_case modules, YAML settings, thin future
scripts, data/raw/processed/splits separation, common predictions/evaluation,
ownership tables, evidence-based README, and explicit reproduction handoffs.
Use the requested uppercase project-guide filenames and six Member placeholders.

Project 2 needs word-aligned punctuation/casing and a shared CRF feature boundary,
not document-classification taxonomy, voting or multilingual translation modules.
Use the preferred smaller `pipeline.py`, `artifacts/models/`, `experiments/` and
top-level `presentation/` layout. No NALTRA code, data, dependency pins, contributor
names or results were copied. Keep all downloaded datasets outside Git here,
even though NALTRA tracks selected compressed data releases.
