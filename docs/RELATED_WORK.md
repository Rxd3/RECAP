# Related work

Every entry below was checked online on 2026-10-30: title, authors, year and venue were read on the
linked page. Each entry ends with how RECAP differs.

## Dataset

**Liu and Beigi (2024).** Xing Yi Liu, Homayoon Beigi. *Spontaneous Informal Speech Dataset for
Punctuation Restoration.* arXiv:2409.11241 (Recognition Technologies, Inc. technical report
RTI-20240917-01). <https://arxiv.org/abs/2409.11241>
Introduces SponSpeech: informal podcast-style speech from Creative Commons YouTube videos, with
creator-written transcripts and a harder test-amb split; it benchmarks full stop, comma and
question mark with text and audio models (BERT reaches overall F1 75.4 on test).
*RECAP differs:* we keep the transcripts only, relabel them into 4, 8 and 12-class token views,
replace the official splits with leak-free primary lists, and train classical models. Our scores use
different gold labels, splits and metrics, so they are not comparable with the paper's.

## Sequence models for punctuation

**Lafferty, McCallum and Pereira (2001).** John D. Lafferty, Andrew McCallum, Fernando C. N. Pereira.
*Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.*
ICML 2001, pp. 282–289. <https://mlanthology.org/icml/2001/lafferty2001icml-conditional>
Introduces CRFs as discriminative sequence models that avoid the label bias of HMM-style models.
*RECAP differs:* we use the linear-chain CRF as the main model and compare it with an HMM on
identical tokens and splits.

**Gotoh and Renals (2000).** Yoshihiko Gotoh, Steve Renals. *Sentence Boundary Detection in Broadcast
Speech Transcripts.* Proc. ISCA ITRW on Automatic Speech Recognition (ASR2000), Paris, pp. 228–235.
<https://www.research.ed.ac.uk/en/publications/sentence-boundary-detection-in-broadcast-speech-transcripts/>
Finite-state models combine an n-gram language model with a pause-duration model to find sentence
boundaries in broadcast news.
*RECAP differs:* our HMM baseline is text-only, with no prosody, and predicts every punctuation
class rather than only boundaries.

**Lu and Ng (2010).** Wei Lu, Hwee Tou Ng. *Better Punctuation Prediction with Dynamic Conditional
Random Fields.* EMNLP 2010, pp. 177–186. <https://aclanthology.org/D10-1018/>
Dynamic CRFs jointly predict sentence type and punctuation in conversational speech transcripts
(IWSLT, English and Chinese) without prosodic cues, and beat a linear-chain CRF.
*RECAP differs:* we keep a linear-chain CRF and get joint prediction from a product label set
(punctuation × casing, mode C), and we measure how much of the score comes from the utterance-final
token.

## Truecasing

**Lita et al. (2003).** Lucian Vlad Lita, Abe Ittycheriah, Salim Roukos, Nanda Kambhatla.
*tRuEcasIng.* ACL 2003, pp. 152–159. <https://aclanthology.org/P03-1020/>
A language-model truecaser that restores case in news text and helps downstream tasks such as NER
and MT.
*RECAP differs:* casing is a 3-way token label predicted together with punctuation, and
approximate mixed casing (`iPhone`) is masked from scoring instead of guessed.

## Joint punctuation and casing

**Gravano, Jansche and Bacchiani (2009).** Agustín Gravano, Martin Jansche, Michiel Bacchiani.
*Restoring Punctuation and Capitalization in Transcribed Speech.* ICASSP 2009, pp. 4741–4744.
<https://research.google/pubs/restoring-punctuation-and-capitalization-in-transcribed-speech/>
Restores both in one step with a text-only n-gram language model, and finds that more training data
helps far more than higher n-gram order.
*RECAP differs:* a discriminative CRF with hand-built features, an explicit 12-class joint label set,
and paired bootstrap intervals for every comparison.

**Pahuja et al. (2017).** Vardaan Pahuja, Anirban Laha, Shachar Mirkin, Vikas Raykar, Lili Kotlerman,
Guy Lev. *Joint Learning of Correlated Sequence Labeling Tasks Using Bidirectional Recurrent Neural
Networks.* Interspeech 2017, pp. 548–552, doi:10.21437/Interspeech.2017-1247.
<https://www.isca-archive.org/interspeech_2017/pahuja17_interspeech.html>
One bidirectional RNN learns punctuation and capitalization jointly and beats separate models.
*RECAP differs:* we test the same question with classical models: each mode C model, projected to
punctuation, is compared with its mode A twin on identical validation sequences
([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), H1).

## IWSLT TED-talk benchmarks

**Che et al. (2016).** Xiaoyin Che, Cheng Wang, Haojin Yang, Christoph Meinel. *Punctuation
Prediction for Unsegmented Transcript Based on Word Vector.* LREC 2016, pp. 654–658.
<https://aclanthology.org/L16-1103/>
A DNN or CNN over pre-trained word vectors predicts punctuation on TED talks from the IWSLT data.
*RECAP differs:* spontaneous podcast speech instead of prepared talks, no pretrained embeddings,
and leakage control by source video.

**Tilk and Alumäe (2016).** Ottokar Tilk, Tanel Alumäe. *Bidirectional Recurrent Neural Network
with Attention Mechanism for Punctuation Restoration.* Interspeech 2016, pp. 3047–3051,
doi:10.21437/Interspeech.2016-1517. <https://www.isca-archive.org/interspeech_2016/tilk16_interspeech.html>
A bidirectional recurrent network with attention restores comma, period and question mark on
English (IWSLT2011 TED) and Estonian. For English it maps exclamation marks and semicolons to
periods, and colons and dashes to commas.
*RECAP differs:* our mode A uses the same projection (D5), but we also keep the 8-class view
(mode B) so the extended marks are not lost.

**Courtland, Faulkner and McElvain (2020).** Maury Courtland, Adam Faulkner, Gayle McElvain.
*Efficient Automatic Punctuation Restoration Using Bidirectional Transformers with Robust
Inference.* IWSLT 2020, pp. 272–279. <https://aclanthology.org/2020.iwslt-1.33/>
A fine-tuned RoBERTa model with prediction aggregation across windows reaches F1 83.9 on the TED
benchmark.
*RECAP differs:* CPU-only classical models with no pretrained weights, on informal speech, with the
utterance-final shortcut reported separately.

**Alam, Khan and Alam (2020).** Tanvirul Alam, Akib Khan, Firoj Alam. *Punctuation Restoration
using Transformer Models for High-and Low-Resource Languages.* W-NUT 2020, pp. 132–142,
doi:10.18653/v1/2020.wnut-1.18. <https://aclanthology.org/2020.wnut-1.18>
Transformer models with data augmentation for English and Bangla punctuation restoration.
*RECAP differs:* no pretrained models; our extension is the joint casing task and the leakage audit.

## Summary: where RECAP sits

The recent benchmark papers above use neural models, mostly pretrained transformers, on TED talks
with three marks. RECAP instead studies
classical sequence labelers (HMM, CRF) on spontaneous speech, adds an 8-class view, a joint
12-class punctuation and casing view, leak-free splits and a measured end-of-utterance shortcut.
