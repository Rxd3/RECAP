# Presentation workspace

Owner: Member 6, with technical input from every member.

The team makes the slide deck after the repository freeze, so no deck is committed. The deck is
built from:

- [PRESENTATION_OUTLINE.md](../docs/PRESENTATION_OUTLINE.md): four 30-second slides with numbers
  taken from Member 5's verified evaluation;
- [EVALUATION_RESULTS.md](../docs/EVALUATION_RESULTS.md) (HMM and shortcut) and
  [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) (CRF and ablations, validation);
- [HMM_RESULTS.md](../experiments/HMM_RESULTS.md) and [CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md)
  for model settings.

Rules: label every number with its split and mode. Keep CRF numbers marked "validation" until the
CRF test predictions are scored. Never rank mode B against mode C.

Generated figures go under the ignored `presentation/assets/`. Every result graphic must trace to
an evaluator output file.
