# CRF Feature Engineering & Ablation Plan — Member 3 Workstream

**Module:** `src/punct_restore/features/crf_features.py`
**Test Suite:** `tests/test_crf_features.py` (23 passed)
**Status:** Architecture and interface finalized; synthetic unit tests verified; integrated with CRFModel and training CLI. Ablation experiments pending verified dataset distribution.

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

**Planned Controlled Comparison:**
1. **Full Improved CRF:** Evaluated across the entire test set.
2. **Boundary-Ablated CRF:** Trained and evaluated without utterance-boundary indicators.
3. **Cross-Check with Member 5 View:** Member 5's evaluation suite implements a `non_final` view that scores only internal tokens $t < T - 1$. We will evaluate our model against Member 5's `non_final` metric to report clean, shortcut-free F1 scores.

---

## 4. Planned Ablation Matrix (Controlled Factors)

| Experiment ID | Feature Configuration | Factor Tested | Baseline Comparison |
| :--- | :--- | :--- | :--- |
| `ABL-01` | `basic` | Baseline (window $\pm 1$, affixes, shape) | Reference |
| `ABL-02` | `basic` + window $\pm 2$ | Impact of context window expansion | vs `ABL-01` |
| `ABL-03` | `basic` + token bigrams | Impact of phrase collocations | vs `ABL-01` |
| `ABL-04` | `basic` + question indicators | Question F1 recovery | vs `ABL-01` |
| `ABL-05` | `basic` + conjunctions & discourse | Comma F1 recovery | vs `ABL-01` |
| `ABL-06` | `improved` (full) | Cumulative combination | vs `ABL-01` |
| `ABL-07` | `improved` w/o boundaries | Utterance-final shortcut isolation | vs `ABL-06` |
