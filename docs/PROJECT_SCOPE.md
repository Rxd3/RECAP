# Project scope

RECAP — Restoring English Casing and Punctuation.
COE025 Group Project 2: Extended Punctuation and Truecasing Restoration Using
Sequence Labeling. Six members; Python; one independent repository.

## Current deliverable

Member 1's corrected SponSpeech release and Member 2's supervised HMM baseline
are complete and verified. The HMM implements log-space Viterbi, unknown words,
persistence, release-aware training, validation selection and prediction export.
Real-data scores, runtime and independent refit evidence are recorded in
[HMM results](../experiments/HMM_RESULTS.md). The study uses Member 5's pinned
shared scorer in a separate checkout. CRF, feature ablations and integrated text
reconstruction remain outside Member 2's completed baseline deliverables.

## Future research scope

Restore a punctuation boundary after each English word and casing of each word.
Compare HMM + Viterbi, a basic linear-chain CRF, and that CRF with improved
features on fixed, shared splits. Support A (4 punctuation), B (8 punctuation),
and C (4 punctuation × 3 casing = 12 joint labels) through configuration.
Investigate clean text and a separately verified ASR condition. Future separate
casing models are a research question; no new label types are introduced here.

## Limits and evidence

Do not claim ready-made extended annotations, source compatibility, dataset
sizes, trained models, or measured gains before verification. BiLSTM-CRF, GPU
frameworks, pretrained model downloads and a finished interactive UI are outside
initial setup. The user separately authorized creating and publishing the Project 2
repository under their account. NALTRA remains a read-only reference. Deadlines
and personal names remain unassigned.

Success later means reproducible data, comparable within-task experiments,
honest error analysis, an integrated demo, and an evidence-based presentation.
