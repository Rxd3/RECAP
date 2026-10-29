# Experiment planning and records

Owners: Members 2–5 produce records; Member 6 maintains reproduction instructions.
Member 2's completed study is recorded in [HMM_RESULTS.md](HMM_RESULTS.md);
[HMM_RUNBOOK.md](HMM_RUNBOOK.md) gives exact reproduction and packaging commands.
Other model notes remain plans until their experiments finish. Generated run
bundles belong under ignored `runs/<run_id>/`.

Copy [experiment.example.yaml](../configs/experiment.example.yaml) when the
pipeline exists. Record the hypothesis, changed variable, full resolved config,
seed, data release and split hashes, code commit, dependency lock, hardware,
commands, artifact/prediction locations and hashes, actual duration, validation
selection criterion, and limitations. Freeze selected settings before final test
evaluation. Preserve negative findings. For ablations hold splits, decoding,
training settings, and runtime measurement conditions constant where possible.

Do not create empty score columns that look like measured zeroes. Keep unfinished
results explicitly pending in prose. The same CRF implementation powers basic
and improved feature conditions.
