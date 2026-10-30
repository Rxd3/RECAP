# Project Presentation Outline & Slide Evidence

**Owner**: Member 6 (Abdullah Jamal)

## Slide Structure

1. **Title & Problem Statement**: Sequence labeling for punctuation/casing restoration on speech transcripts.
2. **Dataset & Preprocessing (Member 1)**: SponSpeech v2 corpus (`sponspeech-m1-v2`, 142k training sequences / 4.14M tokens across Modes A, B, and C).
3. **Baseline Architecture (Member 2)**: HMM with Viterbi decoding.
   - **Untuned Validation Macro F1**: Mode A = 31.95, Mode B = 15.25, Mode C = 39.48 (outperforming shortcut baseline 28.21 / 12.14 / 18.53).
4. **Main CRF Model & Features (Members 3 & 4)**: 6 full-data CRF models trained across 3 label modes.
   - **Feature Sets**: Basic (context ±1, word shape) vs. Improved (context ±2, bigrams, question/discourse markers).
   - **Handoff Status**: Prediction files (`recap_m3_handoff_eval.zip`) delivered with 100% integrity check pass rate (211/211 automated tests passing).
5. **Evaluation & Metrics (Member 5)**: Non-'O' Macro F1 scores, precision, recall, confusion matrices, and joint evaluation mask scoring for Mode C.
6. **Pipeline Integration & Demo (Member 6)**: Live CLI inference (`scripts/run_demo.py`), reproducible setup (`pyproject.toml`), and end-to-end test contract validation (117/117 automated tests passing).