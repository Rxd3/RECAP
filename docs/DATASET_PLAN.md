# Dataset inspection and preparation plan

Owner: Member 1; Member 5 reviews leakage/split/evaluation decisions, Member 6
reviews reproducibility. No dataset has been downloaded. No sizes, final splits,
or label distributions have been calculated.

## Proposed sources and limited page inspection

| Source | Intended role | Setup observations, not an approved configuration |
| --- | --- | --- |
| [IWSLT/iwslt2017](https://huggingface.co/datasets/IWSLT/iwslt2017) | Primary text for deriving extended punctuation and casing | Page describes translation text, shows English in translation fields, lists a CC BY-NC-ND license, and reports a script-dependent viewer; exact selected config and permitted use remain to verify |
| [cdminix/iwslt2011](https://huggingface.co/datasets/cdminix/iwslt2011) | Standard reference and potential ASR condition | Page has no dataset card and its viewer reports a dependency/configuration error; labels, license, splits and ASR contents remain unverified |

Pages were inspected on 2026-10-28. No loader script was executed. An available
page is not evidence of working download compatibility. IWSLT 2017 is not
assumed to contain eight-class punctuation annotations. Inspect preservation of
punctuation/case in original English text before deriving labels. Record source
terms and permitted redistribution/derived-data handling after verification;
no license approval is asserted here.

## Provenance and future download instructions

- [ ] Inspect upstream files and card/script, exact dataset revision and selected
  English configuration; avoid combining mirrored English translation copies.
- [ ] Record actual fields/types, available IDs, document/talk boundaries,
  annotations and native split definitions on a small permitted sample.
- [ ] Verify source license/access conditions and reference-benchmark conventions.
- [ ] Choose a supported reviewed download mechanism and compatible dependency
  versions; do not assume a legacy script works with current Datasets.
- [ ] Enter the exact reproducible command, revision, source URLs, checksums,
  attribution and access date here and in `configs/datasets.yaml`.
- [ ] Store immutable source files under `data/raw/<source_revision>/` locally.

**Download command:** pending source/configuration verification. No placeholder
`load_dataset` call is presented as tested instructions.

## Cleaning, tokenization and annotation decisions

- [ ] Document Unicode normalization, whitespace/markup removal and exclusions.
- [ ] Approve tokenizer, contractions, apostrophes, hyphens, numbers, abbreviations
  and preservation of word alignment; define sequence segmentation/window limits.
- [ ] Derive boundary punctuation from original text; decide repeated marks,
  quote/bracket attachment, dash variants, unsupported marks and terminal/leading
  punctuation without inventing extra labels.
- [ ] Approve casing extraction precedence for one-letter/all-cap/mixed-case and
  uncased tokens; define irrecoverable cases and exclusion counts.
- [ ] Extract labels before input lowercasing; validate A/B/C lengths and IDs.
- [ ] Verify and approve the proposed 8→4 collapse against the benchmark.
- [ ] Version all policies and record representative difficult examples.

## Splits and duplicate/leakage prevention

- [ ] Verify native train/validation/test identities and overlap between both
  proposed sources; do not assume different years mean different talks.
- [ ] Audit exact/normalized/near duplicate text and repeated English translations.
- [ ] Group by talk/document; keep related windows/variants in the same split.
- [ ] Agree how to preserve benchmark splits and handle conflicts; no split ratio
  or random seed is selected yet.
- [ ] Freeze sample/group assignments, seeds where used, exclusions and hashes in
  a release manifest. All models/modes consume the same compatible assignments.
- [ ] Fit any train-derived vocabulary/normalizer/statistics only on training.

## Dataset statistics and class distribution report — pending

After preparation, report source and retained/rejected sequence counts, token
counts, length distribution, vocabulary/OOV statistics, duplicate removals and
group counts, separately per split and source. Report every punctuation and
casing class support, non-O proportions and joint-class distribution per mode.
Explain rare/absent labels and limitations. Do not fill this section with
invented values or treat card counts as processed project statistics.

## Clean versus noisy/ASR evaluation — pending

Verify whether actual ASR hypotheses, clean references and alignments exist.
Record their source and label availability. Define insertions/deletions,
word-alignment policy, parent IDs and shared split groups with Member 5. If only
synthetic corruption is feasible, name it synthetic and record generation seeds
and operations; do not claim measured ASR robustness from clean data alone.

## Reproducible release handoff

- [ ] Publish approved policy/configuration versions, source references and terms.
- [ ] Provide validation report, rejected-record audit, duplicate audit and fixed
  split manifests with checksums through an agreed artifact location.
- [ ] Document exact regeneration command, seed, code commit and dependency lock.
- [ ] Supply small permitted fixtures and contract tests in Member 1's PR.
- [ ] Obtain Members 2, 3, 5 and 6 acceptance of the same release; no consumer
  silently resplits, retokenizes or guesses missing casing.
