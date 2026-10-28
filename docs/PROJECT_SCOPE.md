# Project scope

COE025 Group Project 2: Extended Punctuation and Truecasing Restoration Using
Sequence Labeling. Six members; Python; one independent repository.

## Current deliverable

Initial scaffolding only: package folders, intentionally unfinished modules,
configuration templates, common contracts, team board, Git workflow and local
branches. No training algorithms, probability estimation, Viterbi implementation,
feature extraction, scoring, text reconstruction, or data download is included.

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
initial setup. No remote repository creation, push, or NALTRA write is authorized
by this setup. Deadlines and personal names remain unassigned.

Success later means reproducible data, comparable within-task experiments,
honest error analysis, an integrated demo, and an evidence-based presentation.
