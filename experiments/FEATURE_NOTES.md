# CRF Feature Engineering & Ablation Plan — Member 3 Workstream

**Module:** `src/punct_restore/features/crf_features.py`
**Test Suite:** `tests/test_crf_features.py`
**Status:** The feature code, both presets included, was written by Member 3 (Omar) in commit `aa232be` and used for the six full-data CRF models ([CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md)). Member 4's branch `feat/m4-feature-engineering` has no commits beyond the project scaffold ([contributions](../contributions/Mohamed_2309015740_Mokamaleldin.md)). The feature ablations were run on **validation only** by Member 5: [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) §6. `recap run-ablations` reruns them.

---

## 1. Feature Extraction Architecture (Decision D7)

Following `docs/MODEL_CONTRACT.md`, the extraction interface is:
```python
extract_sequence_features(tokens: Sequence[str], *, config: FeatureConfig | None = None) -> list[dict[str, str | int | float | bool]]
```

### Guarantees
- **Strict Length Preservation:** Exactly one dictionary per token.
- **Gold-Free Discipline:** Inputs are strictly lowercased word tokens (`tokens`). No access to gold punctuation (`punct_labels`), gold casing (`case_labels`), or raw un-tokenized text.
- **Deterministic & Immutable:** Functions do not mutate inputs in-place.
- **Shared Model Engine:** A single CRF implementation accepts injected feature functions, ensuring fair comparison between presets.

---

## 2. Feature Sets & Scientific Hypotheses

### A. Basic Feature Preset (`FeatureConfig.basic()`, version `v1-basic`)
A controlled, parsimonious baseline:
1. **Current Token (`token`):** The lowercased word itself.
2. **Character Affixes (`pref1..3`, `suff1..3`):**
   - *Hypothesis:* Affixes capture morphological word classes (e.g., `-ly` adverbs, `-ing` participles, `-tion` nominalizations) that govern punctuation boundaries without requiring a full POS tagger.
3. **Token Shape & Orthography (`len`, `len_bucket`, `is_numeric`, `has_digit`, `has_hyphen`, `has_apostrophe`):
   - *Hypothesis:* Numbers, hyphenated compounds, and contractions exhibit distinct syntactic distributions (e.g., numbers often precede commas or unit nouns; short tokens like *to, of* are functional heads rarely followed by terminal punctuation).
4. **Immediate Context ($\pm 1$ Window):** `[-1:token]`, `[+1:token]`, `[-1:suff3]`, `[+1:pref3]`.
   - *Hypothesis:* Punctuation placement is locally driven by head-complement boundaries and adjacent function words.
5. **Sequence Boundaries:** `BOS`, `EOS`, `[-1:BOS]`, `[+1:EOS]`.

---

### B. Improved Feature Preset (`FeatureConfig.improved()`, version `v1-improved`)
Linguistically motivated extensions targeting specific error modes:

1. **Extended Context Window ($\pm 2$):** `[-2:token]`, `[+2:token]`.
   - *Hypothesis:* Complex compound sentences, relative clauses, and parenthetical phrases span across more than one adjacent word.
2. **Contextual Token Bigrams:** `bigram_prev` ($w_{t-1}\_w_t$) and `bigram_next` ($w_t\_w_{t+1}$).
   - *Hypothesis:* Lexical collocations and multi-word discourse expressions (*in fact*, *of course*, *thank you*, *as well*) act as cohesive lexical units whose boundaries strongly predict commas or periods.
3. **Question Indicators & Syntactic Inversion:**
   - Wh-lexicon flags: `is_wh`, `[-1:is_wh]`, `utt_starts_wh`.
   - Subject-auxiliary inversion: `q_inversion_curr` (auxiliary at $t-1$ followed by subject pronoun at $t$), `q_inversion_next`, `utt_starts_aux`.
   - *Hypothesis:* Distinguishing `QUESTION` from `PERIOD` in unpunctuated text requires structural and lexical cues. Auxiliary-subject inversion (*did you*, *is it*, *can we*) and interrogative pronouns (*what*, *why*, *how*) strongly signal interrogative clauses.
4. **Clause Markers & Discourse Particles:**
   - Coordinating & subordinating conjunctions: `is_conjunction`, `[-1:is_conj]` (*and*, *but*, *because*, *although*, *since*).
   - Discourse adverbs: `is_discourse` (*however*, *therefore*, *meanwhile*, *well*, *actually*).
   - *Hypothesis:* Discourse particles typically require preceding or following commas in English prosody and written style.
5. **Mode C Casing Indicators:**
   - Honorific titles: `is_honorific` (*dr*, *mr*, *prof*).
   - Proper calendar nouns: `is_calendar` (days of the week, months).
   - First-person singular pronoun: `is_pronoun_i` (*i* $\to$ `FIRST_UPPER`).
   - *Hypothesis:* Since input text is strictly lowercased, lexical matching of deterministic proper nouns allows accurate restoration of `FIRST_UPPER` independent of position.
6. **Positional Bucketing:** `pos_bucket` (`start`, `early`, `middle`, `late`, `end`, `single`).

---

## 3. Investigating the Utterance-Final Position Shortcut

### The Problem
In speech corpora such as SponSpeech, segments represent complete speaker turns or acoustic utterances. Consequently:
- The final token of an utterance ($t = T - 1$) is overwhelmingly followed by a terminal mark (`PERIOD` or `QUESTION`).
- A model with an explicit `EOS=True` feature can easily score high token accuracy on sentence boundaries purely by memorizing the end-of-sequence index, rather than learning syntax or discourse structure.

### Scientific Countermeasure & Controlled Ablation Design
To measure the true syntactic generalization of the CRF, we built the ablation toggle:
```python
FeatureConfig(include_utterance_boundaries=False)
```
When set to `False`, all boundary flags (`BOS`, `EOS`, `[-1:BOS]`, `[+1:EOS]`) are suppressed.

**Comparison as run (feature decisions use validation only; the frozen models were scored once on test):**
1. **Full improved CRF:** selected on the primary validation lists (mode A primary 51.18, [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) §5), then scored once on test with and without the utterance-final token (mode A test primary 49.92, `non_final` 36.40; §1, §3).
2. **Boundary-ablated CRF** (`--no-utterance-boundaries`): not run.
3. **Member 5's `non_final` view** scores only internal tokens $t < T - 1$. Mode A test: HMM 9.13, CRF basic 28.30, CRF improved 36.40 (§3), so the CRF does learn punctuation inside the utterance.

---

## 4. Ablations as Run (validation only)

The additive plan `ABL-01`–`ABL-07` (basic plus one group at a time) was not run. It was replaced by a one-group-removed design on the improved preset:
- Member 5 retrained Member 3's code (`aa232be`) with c1 = c2 = 0.1 and 100 iterations.
- Each variant trained on a fixed 50,000-sequence `default_train` sample (seed 2026), with one fit per row.
- Every variant was scored on the primary validation lists.

Change in primary macro F1 when the group is removed (without − with, points; * = paired 95% interval excludes 0). Source: [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) §6.

| Removed group | A | B | C (joint) |
| :--- | ---: | ---: | ---: |
| word pairs (`ngrams`) | −3.59 * | −1.63 * | −1.49 |
| ±2 window (`window2`) | −1.34 * | −0.76 * | −0.53 * |
| question cues | −1.35 * | −0.44 * | −0.40 * |
| clause cues | −0.07 | +0.17 * | −0.29 |
| position buckets | −0.20 | +0.21 * | +0.83 |
| casing cues | +0.01 | −0.04 | −0.08 |

**Findings:**
- Word pairs, the wider window and the question cues help.
- Clause, position and casing cues have small effects of mixed sign.
- Removing the casing cues does not measurably change mode C casing (−0.08).
- Removing the clause cues changes COMMA F1 by +0.03 (A) and +0.05 (B), and `COMMA|LOWER` F1 by −0.18 (−0.34 to −0.03) in C.

Rerun with `recap run-ablations --mode {A,B,C} --out DIR [--sample 50000] [--variants ...]`. The variants are `improved`, `improved_minus_{window2,ngrams,question,clause,casing,position}`, `improved_plus_filler`, `improved_plus_question2`, `gap_improved`, and (mode C only) `case_improved` and `factored`. The filler-cue variant (`improved_plus_filler`, a Member 5 extension) was trained on the same sample and scored on validation: primary macro F1 changes by +0.04 (A), +0.04 (B) and −0.28 (C), with no interval excluding 0. COMMA F1 rises by +0.21 (0.06 to 0.37) in B and `COMMA|LOWER` F1 by +0.25 (0.05 to 0.45) in C. Hypothesis H4 is therefore partly supported for fillers (a small comma gain) and not supported for discourse markers ([EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) §6, Filler cue).
