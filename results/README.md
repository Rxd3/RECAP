# Evaluation outputs

Owner: Member 5; Member 6 integrates presentation evidence. These `predictions/`,
`tables/`, and `figures/` directories contain empty markers only. Member 2's
completed HMM study and raw reports are under ignored
`experiments/runs/hmm-sponspeech-v2-20261029/`; its reviewed summary is
[HMM_RESULTS.md](../experiments/HMM_RESULTS.md).
Generated outputs are ignored; do not fill tables with illustrative scores.

Score a prediction file with `python scripts/evaluate_models.py score ... --out
results/tables/<run_id>.json`. This also writes `<run_id>.md`. See
[the evaluation protocol](../docs/EVALUATION_PROTOCOL.md#scored-sets-masks-views-and-the-shortcut-baseline)
for the full commands.

Future runs use distinct run IDs and record their input release, mode, split
hash, model/configuration hash, code commit, environment, seed, and actual command.
Publish only evidence produced by completed experiments and reviewed against the
[evaluation protocol](../docs/EVALUATION_PROTOCOL.md). Share large prediction
dumps separately. A small verified summary may be deliberately allowlisted in a
later PR after experiments run. The HMM's text summary is tracked under experiments;
large generated outputs remain ignored.
