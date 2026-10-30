# Configuration templates

`label_sets.yaml` defines the exact ordered label vocabularies. Mode names are
`A`, `B`, `C` throughout the repository. Proposed 8-to-4 projection is explicitly
unapproved and must not silently become a benchmark policy.

`datasets.yaml` is an inspection checklist, not a working loader configuration.
Null values mean unverified; they are not defaults that a loader may guess.
`hmm_sponspeech_v2.yaml` freezes the HMM validation search (smoothing and rare-word cutoff grid).

Owners: Member 1 (sources and label policy), Member 5 (metrics), Member 6
(integration); Members 2–4 review model/feature settings. Version shared changes
through discussion and a PR with all affected owners.
