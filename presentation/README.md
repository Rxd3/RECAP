# Presentation workspace

Owner: Member 6, with technical input from every member.

The team makes the slide deck after the repository freeze, so no deck is committed. The deck is
built from:

- [PRESENTATION_OUTLINE.md](../docs/PRESENTATION_OUTLINE.md): four 30-second slides with numbers
  taken from Member 5's verified evaluation;
- [EVALUATION_RESULTS.md](../docs/EVALUATION_RESULTS.md) (HMM and shortcut) and
  [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) (CRF: test results final,
  validation selection in §5, ablations on validation in §6);
- [HMM_RESULTS.md](../experiments/HMM_RESULTS.md) and [CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md)
  for model settings.

Rules: label every number with its split and mode. The main numbers are test; validation numbers
are model selection and must be labelled as such. Never rank mode B against mode C.

Generated figures stay out of Git (`presentation/assets/` is ignored). Every result graphic must
trace to an evaluator output file in `results/scores/`.
