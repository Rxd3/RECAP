# RECAP

## Restoring English Casing and Punctuation

COE025 Natural Language Processing - Group Project 2

Final report | 1 November 2026

| Team member | Student ID |
| --- | --- |
| Almutesim Alakili | 220911336 |
| Raşit İsaoğlu | 2309015397 |
| Omar Alzyod | 2409015412 |
| Mohamed Kamaleldin Eliwa | 2309015740 |
| Abdlaraouf Alkabir | 2309015482 |
| Abdullah Gamal | 2309015819 |

Repository: [Rxd3/RECAP](https://github.com/Rxd3/RECAP). Code: MIT. SponSpeech data and derived transcript excerpts: CC BY-NC 4.0.

## Abstract

RECAP restores punctuation and capitalization in lowercased, unpunctuated English transcripts. We compare the three required course models: a first-order hidden Markov model (HMM), a basic linear-chain conditional random field (CRF), and a CRF with improved contextual features. We construct standard-punctuation, extended-punctuation and joint punctuation/casing views of SponSpeech, using primary evaluation lists filtered for video overlap and duplicate text. On the standard-punctuation test set, macro F1 rises from 35.12 for the HMM to 43.65 for the basic CRF and 49.92 for the improved CRF. Exploratory pretrained BERT and FLAN-T5 models reach 76.25 and 76.76 on the same task. Removing utterance-final positions exposes the HMM's reliance on boundary cues and confirms that the stronger models learn punctuation inside utterances. Extended rare marks remain difficult. Video-group bootstrap intervals, paired model comparisons, validation-only selection and checksummed outputs support the analysis. Neural truecasing and real ASR evaluation were not performed.

## 1. Task and research questions

The input is a fixed sequence of lowercased words with punctuation removed. The model predicts the punctuation mark after each original word. In the joint task it also predicts the word's casing. This framing makes HMM and CRF outputs directly comparable and preserves one label per input token.

An illustrative target for `hello everyone welcome to our presentation` is `Hello everyone, welcome to our presentation.` This example describes the desired task; it is not presented as a model prediction.

| Mode | Target | Label vocabulary |
| --- | --- | --- |
| A | Standard punctuation | O, COMMA, PERIOD, QUESTION (4) |
| B | Extended punctuation | Mode A plus EXCLAMATION, COLON, SEMICOLON, DASH (8) |
| C | Joint punctuation and casing | Mode A x LOWER, FIRST_UPPER, ALL_UPPER (12) |

The main questions are whether improved CRF features help, whether model gains survive removal of the utterance-final shortcut, and how well the models recover rare punctuation and casing. The exploratory neural study asks how fine-tuned pretrained models compare on punctuation modes A and B. It does not replace the course-baseline comparison.

## 2. Data and preprocessing

### 2.1 Source and label construction

SponSpeech contains informal English speech transcripts from YouTube and supplies punctuation and casing information [1]. RECAP uses reference transcripts only, without audio, prosody or real speech-recognition output. Our canonical views derive all three modes from the same token positions. Input casing is removed after labels have been extracted, so inference cannot inspect gold capitalization.

Mode A is the documented projection of mode B: EXCLAMATION and SEMICOLON map to PERIOD, while COLON and DASH map to COMMA. The joint label `PERIOD|FIRST_UPPER` means that the current word starts with an uppercase letter and is followed by a period. Approximate mixed-case words are masked from primary casing/joint metrics while their punctuation positions remain scored. Unresolved records are retained in full data views with flags and excluded from the default training and primary evaluation lists under the release policy.

### 2.2 Primary splits and leakage control

All primary comparisons use release `sponspeech-m1-v2` and its committed ID lists. The following are the accepted training and primary evaluation sequence counts, rather than the larger official split sizes.

| Mode | Train | Validation | Test | test_amb |
| --- | --- | --- | --- | --- |
| A | 142,722 | 20,040 | 12,321 | 12,734 |
| B | 142,722 | 20,040 | 12,321 | 12,734 |
| C | 141,745 | 19,936 | 12,276 | 12,704 |

The A/B training lists contain 4,137,833 words. Their primary validation contains 618,771 words and 218 video groups; primary test contains 407,750 words and 153 video groups. Test and test_amb are reported separately because they still share 25 source videos.

The original released test has 4,716 of 17,695 utterances (26.7%) from videos that also occur in train or validation. Primary lists remove cross-split video overlap, exact duplicates and near duplicates defined by the approved six-gram rule. This controls leakage from the supervised training splits; it cannot exclude overlap with the pretrained models' much larger corpora. Data policies and complete counts are documented in `DATASET_PLAN.md`, `DATASET_CARD.md` and `DATA_CONTRACT.md`.

## 3. Models and implementation

### 3.1 Required course models

**HMM.** A supervised first-order HMM models label transitions and word emissions, with Viterbi finding the best label sequence. Additive smoothing and rare-word handling limit sparsity. Nine validation candidates per mode choose smoothing 0.01; rare-word cutoff is 1 in A/B and 2 in C.

**Basic CRF.** A linear-chain CRF [2], trained with CRFsuite, conditions labels on observable token features. The basic preset uses a +/-1 context window, affixes, token shape and utterance boundaries.

**Improved CRF.** The same CRF adds a +/-2 window, word pairs, question cues, clause/discourse cues, position features and casing-related cues derived only from lowercased input and observable context. Both CRF presets use fixed c1 = c2 = 0.1 and a maximum of 100 L-BFGS iterations. These settings were not selected by a hyperparameter search, and every delivered fit reached the iteration cap.

### 3.2 Exploratory classical extensions

Threshold decoding predicts a non-O label when 1 - P(O) exceeds a validation-selected threshold. A later prior-adjusted extension also divides each non-O class probability by its training frequency raised to gamma, with threshold and gamma jointly selected on validation. An additional hidden-event n-gram language model treats punctuation as events between words and uses exact Viterbi decoding. These follow-up comparisons were added after the first test report and are identified as exploratory even though their settings were chosen on validation.

Controlled validation studies also examine feature-group ablations, filler cues, question lookback, a factored punctuation/casing pipeline and word clusters. Most of these studies train on a shared 50,000-sequence sample; their absolute scores are not substituted for the full-data course results.

### 3.3 Pretrained neural comparison

**BERT.** `bert-base-uncased` [3] is fine-tuned as a word-label classifier. Only the first complete subword receives supervision; special tokens and later pieces are ignored in the loss. Inputs are limited to 512 pieces. Cut or partly retained words receive default O predictions. Both selected BERT runs record zero cut-off validation words.

**FLAN-T5.** `google/flan-t5-base` combines the T5 text-to-text architecture [4] with instruction tuning [5]. It generates reconstructed text from the input words. The output is aligned to the original word sequence with whitespace processing and SequenceMatcher matching blocks. Insertions create no additional input labels; unmatched or changed input words get default O labels. Generation uses four beams and at most `2 * input_words + 16` new tokens per utterance. Source and target training sequences are dynamically padded without truncation.

Separate models are trained for A and B. Neural mode C was omitted for time before its training began, so there is no neural casing or joint comparison. Both neural families use the complete training ID lists.

## 4. Selection, metrics and uncertainty

The HMM settings and neural checkpoints are chosen on primary validation. BERT evaluates both learning rates 2e-5 and 5e-5 after epochs 1, 2 and 3; FLAN-T5 evaluates LR 1e-4 after all three epochs. Selection maximizes the official evaluator's recorded primary macro F1. Ties prefer fewer epochs, then the lower learning rate. Four frozen neural selection records and full runtime records are committed at `64fa17e`, before the subsequent held-out inference driver.

The primary score is macro F1 over the three non-O punctuation classes in A and seven non-O classes in B. C averages all 12 joint classes on unmasked positions. Undefined class precision, recall or F1 is zero; unsupported classes remain in the macro average. Accuracy and micro F1 are supplementary because O is 87.3% of A/B training tokens. Macro scores across modes measure different tasks and are not ranked against each other.

We report 95% percentile intervals from 1,000 bootstrap resamples of source-video groups, seed 2026. Paired comparisons use the same resampled videos for both models. Intervals reflect evaluation-sample variation, not variation across independent training seeds. Neural comparisons are exploratory and are outside the report's Holm correction families. The established HMM/CRF final evaluations used frozen models; Member 2 reports that each neural held-out list was predicted once. The verification in section 8 distinguishes this provenance claim from what can be rechecked using shared files.

### 4.1 Selected neural settings and runtime

| Run | Learning rate | Epoch | Training minutes: selected / search | Validation ms/utterance |
| --- | --- | --- | --- | --- |
| BERT A | 2e-05 | 3 | 64.08 / 125.05 | 1.8463 |
| FLAN-T5 A | 1e-04 | 3 | 206.94 / 206.94 | 154.6956 |
| BERT B | 5e-05 | 3 | 60.18 / 121.85 | 1.7876 |
| FLAN-T5 B | 1e-04 | 2 | 135.61 / 204.06 | 191.9346 |

Runtime records use the RTX 5070 (12,227 MiB VRAM), Windows, Python 3.13.14, torch 2.9.1+cu128 and transformers 4.57.3. All runs use seed 2026, AdamW, batch size 4, gradient accumulation 4 and gradient checkpointing. BERT uses fp16 and FLAN-T5 bf16. Training time excludes validation and checkpoint writing. Validation latency includes tokenization, inference/generation and alignment, excludes model/data loading, and takes the median of three full passes after one full warm-up.

On this common GPU validation protocol, BERT uses about 1.8 ms per utterance versus 155-192 ms for four-beam FLAN-T5. These timings are not directly comparable with CPU measurements of the classical models. FLAN-T5 has 319/618,771 unaligned validation words in A (0.0516%) and 291/618,771 in B (0.0470%). These diagnostics describe validation only.

## 5. Final results

### 5.1 Course baselines and classical follow-ups

Primary test macro F1 %, with 95% video-bootstrap intervals. Each column is a separate task. The first three rows are the required course models; the remaining rows are exploratory comparisons.

| Model | A: punctuation | B: extended punctuation | C: joint labels |
| --- | --- | --- | --- |
| HMM + Viterbi | 35.12 (32.80-37.31) | 16.79 (15.28-18.49) | 37.41 (34.56-40.71) |
| CRF basic | 43.65 (41.74-45.59) | 20.25 (18.87-21.95) | 43.85 (42.13-45.47) |
| CRF improved | 49.92 (47.91-52.01) | 22.82 (21.42-24.42) | 47.02 (44.97-49.00) |
| Hidden-event LM (exploratory) | 48.38 (46.21-50.62) | 22.51 (21.08-23.98) | 41.18 (39.42-42.70) |
| CRF + prior-adjusted threshold (exploratory) | 51.34 (49.61-53.06) | 24.25 (23.06-25.54) | 49.77 (47.26-51.95) |

The improved CRF exceeds the HMM by 14.80, 6.03 and 9.61 points in A, B and C, respectively. Its gains over the basic CRF are 6.27, 2.57 and 3.17 points. All nine paired intervals covering the three course-model comparisons in each mode exclude zero, and the ranking holds on test_amb. Prior adjustment adds 1.42, 1.43 and 2.74 points over the plain improved CRF on test, with paired intervals excluding zero; this is a decoding follow-up, not a fourth required course model.

### 5.2 Neural punctuation results

Validation selects the checkpoint; test and test_amb measure held-out performance. Scores are macro F1 %, with 95% video-bootstrap intervals for both held-out columns.

| Model | Mode | Validation | Test (95% CI) | test_amb (95% CI) |
| --- | --- | --- | --- | --- |
| BERT | A | 76.72 | 76.25 (74.51-77.90) | 75.46 (73.81-76.97) |
| FLAN-T5 | A | 77.17 | 76.76 (74.78-78.38) | 75.45 (73.69-77.10) |
| BERT | B | 42.39 | 36.43 (34.41-38.26) | 36.90 (35.31-38.30) |
| FLAN-T5 | B | 38.81* | 36.79 (34.32-39.28) | 36.31 (33.65-38.88) |

*FLAN-T5 B validation is the official selected-candidate score in the committed frozen record. Its validation predictions were not in the public packet and were not independently rescored for this report. The other three validation files were rescored exactly, including all bootstrap intervals.

The neural models improve substantially over the full-data improved CRF within each punctuation mode. The following are paired test differences, neural minus improved CRF, in percentage points.

| Model | Mode | Test gain | Paired 95% CI |
| --- | --- | --- | --- |
| BERT | A | +26.33 | 24.18 to 28.63 |
| FLAN-T5 | A | +26.84 | 24.63 to 29.23 |
| BERT | B | +13.61 | 11.80 to 15.27 |
| FLAN-T5 | B | +13.97 | 11.60 to 16.05 |

Both pretrained models exceed the prior-adjusted CRF as well: by 24.91-25.42 points in A and 12.18-12.54 in B. Their paired intervals exclude zero. FLAN-T5 has a slightly higher point estimate on test in each mode, but there is no reported paired BERT-versus-T5 interval; these small differences do not establish a statistically reliable winner. BERT provides much lower validation latency under the recorded generation settings.

## 6. What the results show

### 6.1 Utterance boundaries account for much of the HMM score

Every canonical A/C sequence ends in PERIOD or QUESTION, and the default/primary B lists have the same pattern. A rule that predicts only a final mark can therefore look useful while missing internal punctuation. The following table scores mode A test with and without each utterance's last token.

| Model | All positions | Non-final positions |
| --- | --- | --- |
| HMM | 35.12 | 9.13 |
| CRF basic | 43.65 | 28.30 |
| CRF improved | 49.92 | 36.40 |
| BERT | 76.25 | 66.65 |
| FLAN-T5 | 76.76 | 67.70 |

The HMM drops from 35.12 to 9.13. The improved CRF retains 36.40, BERT 66.65 and FLAN-T5 67.70. The stronger models' improvements therefore extend inside utterances. The original shortcut baseline reaches 27.48 overall but zero on non-final punctuation.

### 6.2 Frequent punctuation benefits most; rare marks remain difficult

On mode A test, the improved CRF obtains COMMA 52.52, PERIOD 71.02 and QUESTION 26.23 F1. BERT reaches 70.64, 82.16 and 75.97; FLAN-T5 reaches 70.39, 82.83 and 77.07. The largest neural gain is on QUESTION. Paired neural-versus-CRF intervals exclude zero for all three classes.

Mode B remains difficult because its macro average weights rare marks equally. On test, BERT and FLAN-T5 achieve EXCLAMATION F1 of only 2.61 and 2.87, and SEMICOLON F1 of 0.00 and 3.15. COLON is 18.12 and 20.69. Strong common-mark scores do not imply reliable recovery of every extended mark. Some COLON gold labels also reflect leftover speaker prefixes.

### 6.3 Features, casing and the original hypotheses

Validation ablations identify word pairs, the wider context window and question cues as useful feature groups in A/B. Filler cues produce small COMMA gains without a clear macro gain. Joint training helps punctuation on non-final positions, but the original hypothesis that it consistently raises PERIOD F1 is not supported. Plain threshold decoding primarily raises comma recall rather than reliably improving the rare punctuation classes.

In mode C test, masked casing macro F1 is 79.81 for the HMM and 85.09 for the improved CRF. A later controlled 50k-sample validation study finds a factored punctuation/casing pipeline 2.08 points above a joint CRF on the joint metric, largely through better casing; this is a validation finding, not a replacement final-test result. Full H1-H6 outcomes and their supports are retained in `EVALUATION_RESULTS_CRF.md`.

### 6.4 Leakage and simulated word errors

In the supplementary mode A analysis, the improved CRF scores 67.09 on official-test utterances from training-set videos, versus 49.50 on the leakage-controlled kept pool. The HMM scores 33.54 versus 34.74. The 49.50 kept-pool result uses 12,668 utterances and is different from the 49.92 result on the stricter 12,321 primary IDs; those denominators are not interchanged.

An exploratory synthetic-noise study applies a substitution-dominant word-error mix at 5%, 10% and 15% WER, using three random seeds. At 15% WER, mean mode A scores move from 35.12 to 33.82 for the HMM, 43.65 to 39.30 for the basic CRF, and 49.92 to 43.46 for the improved CRF. The course-model ranking holds. These are simulated corruptions, not real ASR results, and the neural models were not evaluated under this noise condition.

## 7. Limitations

- CRF regularization and iteration settings are fixed, and all fits reached the 100-iteration cap. The comparison is between the delivered models, not exhaustively tuned architectures.
- Each setting has one training seed. Bootstrap intervals quantify variation across evaluation videos, not training reproducibility across seeds.
- Speaker labels, subtitle cue numbers and imperfect punctuation remain in the gold data. Rare-class support and gold quality limit strong claims, especially for mode B COLON and the rare mode C joint labels.
- Supervised splits are filtered for video/text overlap, but channel overlap and unknown pretrained-corpus overlap remain possible. Neural gains combine architecture with substantial external pretraining.
- Text-to-text generation may change words; alignment defaults penalize those edits. BERT's 512-piece limit can lose tail labels on sufficiently long inputs, although the recorded A/B validation runs have no cut-off words.
- Modes have different target vocabularies. No neural mode C, neural casing evaluation, real-ASR condition or neural synthetic-noise result is claimed.
- Exploratory additions follow the initial course-model analysis. Their paired intervals are not included in the Holm families. Small BERT-versus-FLAN-T5 differences are not tested as a paired comparison here.

## 8. Verification and reproducibility

### 8.1 Checks completed for this report

An independent audit on 1 November 2026 checks 41 committed score reports, including all eight neural held-out evaluations, against their stored confusion matrices. Per-class precision/recall/F1, macro F1, micro F1, accuracy, primary values and token totals agree. Every gold, primary-ID and mode C mask identity agrees with the local v2 release. The 24 neural paired reports agree with the corresponding prediction identities and score differences and record the required bootstrap settings.

The public Drive validation archive has 588,758 bytes and SHA-256 `ef89cdd777c22c2dfab6368f4406d6dc5f289087aca3c4f675c8b5ddda252cbe`. Its eight entries pass CRC and all entry checksums. The available BERT A/B and FLAN-T5 A validation predictions and manifests match the committed score-file identities. Re-running the unchanged official evaluator with 1,000 video-group bootstrap resamples reproduces every reported block, count and interval exactly for these three files. Corresponding selection records, training manifests and artifact metadata in the shared model archive match their committed frozen records and manifest references. All four frozen candidate grids and selected winners are internally consistent.

### 8.2 Evidence boundary and artifact access

The inspected public Drive packet is an interim validation handoff. It does not contain test/test_amb predictions, their manifests, exclusive inference receipts or FLAN-T5 B validation predictions. The model archive contains selected BERT A/B and FLAN-T5 A artifacts. FLAN-T5 B weights and the final 12-pair prediction handoff were not present in the inspected public folders. The final held-out scores in this report are Member 5's committed evaluations. Their arithmetic and input identities have been independently checked; their video-bootstrap draws have not been rerun from missing held-out predictions. Once-only inference remains attributed to the producing workflow and Member 2, rather than independently certified from receipts. Tensor weight files were not downloaded and independently rehashed in this audit.

The compact evidence is committed in `results/scores/neural_verification/verification.json` and `selection_summary.json`. Existing HMM/CRF data and model access is described in `REPRODUCIBILITY.md`. Shared files remain at [team Drive](https://drive.google.com/drive/folders/1ZKDNnRHXsLJf3UjOdKVgQ0bYpo12IQfH); raw data, predictions and weights are excluded from Git. Score reports record input SHA-256 values.

To re-score an available A/B prediction file, use `scripts/evaluate_models.py score` with its matching v2 gold file, primary ID list and prediction manifest, plus `--bootstrap 1000 --seed 2026`. Mode C additionally requires its split-specific casing mask. Existing output files are not overwritten. Re-scoring existing predictions does not require repeating model inference. All detailed reproduction commands and package versions remain in `REPRODUCIBILITY.md`.

## 9. Conclusion

The required course comparison supports a consistent result: improved CRF features outperform the basic CRF, and both outperform the HMM within every mode. Boundary-controlled evaluation shows why this matters: the HMM relies heavily on utterance endings, whereas the improved CRF and pretrained models recover substantial punctuation inside utterances. The exploratory neural models provide the highest A/B test scores, with BERT offering much lower validation latency than four-beam FLAN-T5 under the recorded GPU protocol. Extended rare marks, gold noise, pretraining comparability and missing shared reproduction artifacts remain practical limits. The reported claims are tied to named splits, fixed label definitions and inspectable evaluation outputs.

## References

1. Liu, X. Y., and Beigi, H. (2024). *Spontaneous Informal Speech Dataset for Punctuation Restoration*. Recognition Technologies, Inc. technical report. [arXiv:2409.11241](https://arxiv.org/abs/2409.11241).
2. Lafferty, J., McCallum, A., and Pereira, F. (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*. ICML. [Paper](https://mlanthology.org/icml/2001/lafferty2001icml-conditional).
3. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT. [ACL Anthology](https://aclanthology.org/N19-1423/).
4. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR, 21(140), 1-67. [Paper](https://jmlr.org/papers/v21/20-074.html).
5. Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416).
6. RECAP team (2026). *Evaluation results, dataset documentation and frozen neural selections*. [Repository](https://github.com/Rxd3/RECAP); [frozen neural records, 64fa17e](https://github.com/Rxd3/RECAP/blob/64fa17e8c92289d14b7f6a07fc1f1b703365c6c9/experiments/NEURAL_NOTES.md).
