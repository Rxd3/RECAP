# Simulated ASR Noise Robustness (`v2` Edit Model and Seed Variance)

## 1. Method

This study extends RECAP's exploratory simulated-ASR robustness check (`docs/EVALUATION_RESULTS_CRF.md` §13) with a substitution-dominant edit model (`--edit-model v2` in `scripts/simulate_asr_noise.py`, implemented in `src/punct_restore/data/asr_noise.py`) and evaluates seed variance across three random seeds (2026, 2027, 2028). The original `v1` edit model remains the default (`--edit-model v1`) and produces byte-identical outputs.

### Edit operations in `v2`

- **Confusion substitution (cost 1):** replaces a token with a different word at character Levenshtein edit distance 1 that shares the same first letter. Candidate words come exclusively from the `default_train` split vocabulary with a minimum count of 5 (never from `validation` or `test`). The token keeps its punctuation and casing labels. Homophone substitution (`their/there/they're`, `your/you're`, `its/it's`, `to/too/two`, `then/than`) stays available as a second substitution source.
- **Deletion (cost 1):** deletes a function word or any word of 3 characters or fewer, never the last remaining token of an utterance. As in `v1`, a deleted token's punctuation label moves to the previous surviving token if that token has `O`, and is dropped otherwise.
- **Insertion (cost 1):** inserts a filler or short function word (`um`, `uh`, `the`, `a`, `and`, `so`, `like`, `i`) before a token. The inserted token receives punctuation `O`, casing `LOWER`, mode C casing mask `0` (included in `excluded_positions` so its casing is not scored), and an empty character offset span `[pos, pos]` at the start offset of the following token.
- **Swap (cost 2):** swaps two adjacent non-final tokens as in `v1`, capped at 10% of the error budget (`2 * swaps <= int(0.10 * budget)`).

**Target cost mix and budget:** target cost shares are 45.00% substitution, 30.00% deletion, 15.00% insertion, and 10.00% swap. As in `v1`, the error budget is `round(wer * tokens_before)` across the 407,750 primary test tokens, and edits are drawn with a single seeded `random.Random(seed)` so identical seeds yield identical outputs.

**Evaluation protocol:** each frozen model (`HMM`, `CRF basic`, `CRF improved`) in modes `A`, `B`, and `C` predicted each noisy primary `test` set once without retuning (`v2` at WER 5%, 10%, and 15% across seeds 2026, 2027, 2028; `v1` at WER 15% across seeds 2026, 2027, 2028, reusing `results/scores/asr_noise/wer0.15/` for seed 2026). Every run was scored with `scripts/evaluate_models.py score --bootstrap 1000`, using the remapped mode C casing masks.

## 2. Achieved Edit Mix (`v2` and `v1`)

Cost share % per edit type across seeds 2026, 2027, and 2028 (mean and min–max range across seeds; swaps have cost 2, all other edits cost 1):

| Edit model | Target WER | Achieved WER % | Budget | Tokens before | Tokens after (mean) | Substitution % | Deletion % | Insertion % | Swap % |
|---|---|---|---|---|---|---|---|---|---|
| `v2` target | — | — | — | — | — | 45.00 | 30.00 | 15.00 | 10.00 |
| `v2` | 5.00% | 5.00 | 20,388 | 407,750 | 404,691.0 | 45.00 (45.00–45.00) | 30.00 (30.00–30.00) | 15.00 (15.00–15.00) | 10.00 (10.00–10.00) |
| `v2` | 10.00% | 10.00 | 40,775 | 407,750 | 401,633.0 | 45.00 (45.00–45.00) | 30.00 (30.00–30.00) | 15.00 (15.00–15.00) | 10.00 (10.00–10.00) |
| `v2` | 15.00% | 15.00 | 61,162 | 407,750 | 398,575.0 | 45.00 (45.00–45.00) | 30.00 (30.00–30.00) | 15.00 (15.00–15.00) | 10.00 (10.00–10.00) |
| `v1` | 15.00% | 15.00 | 61,162 | 407,750 | 392,459.0 | 25.00 (25.00–25.00) | 25.00 (25.00–25.00) | 0.00 (0.00–0.00) | 50.00 (50.00–50.00) |

## 3. Primary Macro F1 by Mode (`v2`)

Primary macro F1 % on the primary `test` lists: clean (0% WER) alongside the mean over seeds 2026, 2027, and 2028 with the min–max seed range in parentheses, plus the mean drop from 0% to 15% WER in percentage points.

### Mode A (`punctuation/all`)

| Model | Clean (0%) | WER 5% | WER 10% | WER 15% | Drop (0% → 15%) |
|---|---|---|---|---|---|
| HMM | 35.12 | 34.69 (34.67–34.70) | 34.25 (34.22–34.27) | 33.82 (33.75–33.87) | −1.31 |
| CRF basic | 43.65 | 42.11 (42.04–42.23) | 40.71 (40.59–40.80) | 39.30 (39.13–39.51) | −4.36 |
| CRF improved | 49.92 | 47.53 (47.35–47.64) | 45.34 (45.20–45.52) | 43.46 (43.10–43.88) | −6.47 |

### Mode B (`punctuation/all`)

| Model | Clean (0%) | WER 5% | WER 10% | WER 15% | Drop (0% → 15%) |
|---|---|---|---|---|---|
| HMM | 16.79 | 16.54 (16.52–16.55) | 16.28 (16.25–16.34) | 15.94 (15.84–15.99) | −0.86 |
| CRF basic | 20.25 | 19.50 (19.36–19.60) | 18.71 (18.55–18.96) | 17.97 (17.72–18.26) | −2.28 |
| CRF improved | 22.82 | 21.74 (21.64–21.91) | 20.61 (20.49–20.82) | 19.73 (19.50–20.02) | −3.09 |

### Mode C (`joint/unmasked`)

| Model | Clean (0%) | WER 5% | WER 10% | WER 15% | Drop (0% → 15%) |
|---|---|---|---|---|---|
| HMM | 37.41 | 33.76 (33.65–33.88) | 32.00 (31.74–32.19) | 30.43 (30.14–30.92) | −6.98 |
| CRF basic | 43.85 | 38.49 (38.45–38.57) | 35.45 (35.38–35.58) | 33.23 (33.10–33.46) | −10.62 |
| CRF improved | 47.02 | 41.93 (41.42–42.85) | 38.44 (37.80–39.40) | 35.93 (35.43–36.84) | −11.09 |

## 4. Comparison: `v1` vs `v2` at 15% WER

Primary macro F1 % at 15% WER averaged across seeds 2026, 2027, and 2028 (mean and min–max range) for both edit models, with the mean difference `v2 − v1` in percentage points:

| Mode | Model | Clean (0%) | `v1` 15% WER | `v2` 15% WER | `v2` − `v1` (pts) | `v1` seed range | `v2` seed range |
|---|---|---|---|---|---|---|---|
| A | HMM | 35.12 | 35.17 (35.16–35.19) | 33.82 (33.75–33.87) | −1.36 | 0.03 | 0.13 |
| A | CRF basic | 43.65 | 41.65 (41.52–41.74) | 39.30 (39.13–39.51) | −2.35 | 0.22 | 0.38 |
| A | CRF improved | 49.92 | 45.40 (45.27–45.50) | 43.46 (43.10–43.88) | −1.94 | 0.23 | 0.78 |
| B | HMM | 16.79 | 16.84 (16.80–16.88) | 15.94 (15.84–15.99) | −0.91 | 0.08 | 0.15 |
| B | CRF basic | 20.25 | 19.21 (19.08–19.28) | 17.97 (17.72–18.26) | −1.24 | 0.20 | 0.54 |
| B | CRF improved | 22.82 | 20.73 (20.60–20.87) | 19.73 (19.50–20.02) | −1.01 | 0.27 | 0.52 |
| C | HMM | 37.41 | 37.07 (36.98–37.14) | 30.43 (30.14–30.92) | −6.63 | 0.16 | 0.78 |
| C | CRF basic | 43.85 | 42.00 (41.83–42.15) | 33.23 (33.10–33.46) | −8.77 | 0.31 | 0.36 |
| C | CRF improved | 47.02 | 43.91 (43.61–44.12) | 35.93 (35.43–36.84) | −7.97 | 0.51 | 1.41 |

## 5. Findings

1. **Does the improved CRF still degrade faster than the HMM?** Yes. At 15% WER under `v2`, the improved CRF loses 6.47 points in mode A (49.92 → 43.46), compared with 4.36 points for the basic CRF (43.65 → 39.30) and 1.31 points for the HMM (35.12 → 33.82). The same pattern holds in mode B (improved CRF −3.09 pts vs HMM −0.86 pts) and mode C (improved CRF −11.09 pts vs HMM −6.98 pts). Because the improved CRF relies on wide context windows and exact lexical word-pair features, token corruptions perturb more active features per edit than in the basic CRF or the 1st-order HMM.
2. **Is the ranking stable across seeds?** Yes — `CRF improved > CRF basic > HMM` holds for every individual seed (2026, 2027, 2028) at every WER (5%, 10%, 15%) in all three modes (`A`, `B`, `C`), as well as across all `v1` 15% WER seeds. Seed variance is small: across all 27 `(mode, model, WER)` cells in `v2`, the min–max spread over seeds 2026, 2027, and 2028 is between 0.02 and 1.60 macro F1 points (and between 0.03 and 0.51 points for `v1` at 15% WER), which is an order of magnitude smaller than the gaps between model families.
3. **How do `v1` and `v2` differ?** In edit composition at 15% WER, `v1` spends 50.00% of its error budget on adjacent-word swaps, 25.00% on function-word deletions, and 25.00% on 5-group homophone substitutions, with no insertions. By contrast, `v2` matches the target mix on all three WERs (45.00% substitution, 30.00% deletion, 15.00% insertion, and 10.00% swap at 15% WER). In mode A at 15% WER, replacing half-budget swaps with confusion substitutions, short-word deletions, and filler/function-word insertions shifts mean macro F1 from 35.17 to 33.82 (−1.36 pts) for the HMM, from 41.65 to 39.30 (−2.35 pts) for CRF basic, and from 45.40 to 43.46 (−1.94 pts) for CRF improved (and −7.97 pts for CRF improved in mode C). Unlike `v1`, where the HMM barely moved because non-final swaps left utterance-final transitions intact, `v2` deletions of short words (<= 3 characters) and insertions of fillers/short function words also degrade the HMM; in mode C, confusion substitutions that replace a `LOWER` word with a short token that models capitalise (such as an acronym in `default_train`) also reduce precision on the rare `*|ALL_UPPER` joint classes.

## 6. Limits

- **Synthetic token-level edits, not acoustic ASR:** `v2` corrupts clean reference transcripts uniformly at random among eligible positions rather than decoding audio through an acoustic and language model, where errors cluster around fast speech, rare names, and acoustic confusions.
- **Orthographic proxy for confusion substitutions:** confusion candidates use character Levenshtein distance 1 with the same initial letter from `default_train` (count >= 5) rather than phonetic or subword acoustic confusion matrices.
- **Fixed insertion list and neutral labels:** insertions draw from 8 common fillers and short function words (`um`, `uh`, `the`, `a`, `and`, `so`, `like`, `i`) and always receive `O` punctuation and `LOWER` casing (masked in mode C), so false sentence boundaries induced by multi-word ASR hallucinations are not tested.
- **Single-token edits only:** multi-token phrase substitutions, word merges/splits (e.g., compound words), and timing/prosody cues remain outside the scope of this text-only check.

Files: `results/scores/asr_noise_v2/summary.json`, `results/scores/asr_noise_v2/v2/wer<rate>/seed<seed>/`, and `results/scores/asr_noise_v2/v1/wer0.15/seed<seed>/`.
