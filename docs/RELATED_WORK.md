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
*RECAP differs:* we keep the transcripts only, derive the course's 4-, 8- and 12-class token labels from them,
replace the official splits with leak-free primary lists, and train classical models. Our scores use
different gold labels, splits and metrics, so they are not comparable with the paper's.

## Sequence models for punctuation

**Lafferty, McCallum and Pereira (2001).** John D. Lafferty, Andrew McCallum, Fernando C. N. Pereira.
*Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.*
ICML 2001, pp. 282–289. <https://mlanthology.org/icml/2001/lafferty2001icml-conditional>
Introduces CRFs as discriminative sequence models that avoid the label bias of HMM-style models.
*RECAP differs:* we use the linear-chain CRF as the main model and compare it with an HMM on
identical tokens and splits.

**Stolcke and Shriberg (1996).** Andreas Stolcke, Elizabeth Shriberg. *Automatic linguistic segmentation
of conversational speech.* ICSLP 1996, pp. 1005–1008, doi:10.21437/ICSLP.1996-259.
<https://www.isca-archive.org/icslp_1996/stolcke96_icslp.html>
An n-gram language model over words and boundary events segments conversational speech transcripts using
word-level information only: the hidden-event language model.
*RECAP differs:* RECAP's hidden-event LM (`src/punct_restore/models/hidden_event_lm.py`) follows this
design but predicts every punctuation class of the mode, not only boundaries, and serves as a comparison
method next to the HMM and the CRFs.

**Stolcke et al. (1998).** Andreas Stolcke, Elizabeth Shriberg, Rebecca Bates, Mari Ostendorf, Dilek Hakkani,
Madelaine Plauche, Gokhan Tur, Yu Lu. *Automatic detection of sentence boundaries and disfluencies based on
recognized words.* ICSLP 1998, doi:10.21437/ICSLP.1998-486.
<https://www.isca-archive.org/icslp_1998/stolcke98_icslp.html>
Combines word-based event n-gram language models with decision-tree prosodic models to find sentence
boundaries and disfluencies in recognized Switchboard speech; the combination beats each source alone.
*RECAP differs:* the hidden-event LM uses words only, with no prosody, and labels every punctuation class.

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

## Classical tagging precedents: maximum entropy and CRFs

These are the closest precedents for RECAP's models: punctuation, sentence boundaries or casing treated
as per-word tagging with maximum-entropy or CRF models.

**Huang and Zweig (2002).** Jing Huang, Geoffrey Zweig. *Maximum entropy model for punctuation
annotation from speech.* ICSLP 2002, pp. 917–920, doi:10.21437/ICSLP.2002-307.
<https://www.isca-archive.org/icslp_2002/huang02b_icslp.html>
Treats punctuation insertion as word tagging, with a maximum-entropy tagger over word and pitch/timing
cues, on Switchboard reference and recognizer transcripts.
*RECAP differs:* words only (no prosody), the course's extended and joint label sets, and sequence models
(HMM, CRF) as the main models. RECAP's exploratory gap classifier is a per-token classifier on the CRF's
features; it loses 1.27 / 1.70 / 4.74 points to the improved CRF in modes A / B / C (validation, 50k
sample), a gap that mixes the label transitions with training differences
([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §14.1).

**Liu, Stolcke, Shriberg and Harper (2005).** Yang Liu, Andreas Stolcke, Elizabeth Shriberg, Mary Harper.
*Using Conditional Random Fields for Sentence Boundary Detection in Speech.* ACL 2005, pp. 451–458.
<https://aclanthology.org/P05-1056/>
Compares a CRF with HMM and maximum-entropy classifiers for sentence boundaries in conversational and
broadcast speech, on reference and ASR transcripts; the CRF has the lowest error.
*RECAP differs:* the same HMM-versus-CRF comparison, but for every punctuation class and for casing,
from words only.

**Batista, Caseiro, Mamede and Trancoso (2008).** Fernando Batista, D. Caseiro, N. Mamede, Isabel
Trancoso. *Recovering capitalization and punctuation marks for automatic speech recognition: Case study
for Portuguese broadcast news.* Speech Communication 50, pp. 847–862, doi:10.1016/j.specom.2008.05.008.
<https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.specom.2008.05.008>
Restores both capitalization and punctuation in Portuguese broadcast news: finite-state transducers and
maximum-entropy models for capitalization, and a maximum-entropy model with lexical and acoustic
features for punctuation.
*RECAP differs:* English spontaneous speech and words only. RECAP's main mode C model is one joint
12-class tagger, but an exploratory factored pipeline (a punctuation CRF plus a casing-only CRF, separate
models as here) beats it on the joint metric by +2.08 (1.22 to 2.94) and on masked casing by +1.49, while
the joint model keeps an edge on punctuation inside the utterance (−0.83 for the factored pipeline)
(validation, 50k sample; [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §14.5).

**Wang, Ng and Sim (2012).** Xuancong Wang, Hwee Tou Ng, Khe Chai Sim. *Dynamic conditional random
fields for joint sentence boundary and punctuation prediction.* Interspeech 2012, pp. 1384–1387,
doi:10.21437/Interspeech.2012-398. <https://www.isca-archive.org/interspeech_2012/wang12h_interspeech.html>
A dynamic CRF with word-level, prosodic and n-gram features predicts sentence boundaries and punctuation
jointly on TDT3 English broadcast news, beating two-stage linear-chain CRF or maximum-entropy pipelines.
*RECAP differs:* a single linear-chain CRF, words only, joint with casing rather than with sentence type.

**Ueffing, Bisani and Vozila (2013).** Nicola Ueffing, Maximilian Bisani, Paul Vozila. *Improved models
for automatic punctuation prediction for spoken and written text.* Interspeech 2013, pp. 3097–3101,
doi:10.21437/Interspeech.2013-675. <https://www.isca-archive.org/interspeech_2013/ueffing13_interspeech.html>
CRFs over language-model scores, token n-grams, sentence length and parse-tree features predict
punctuation in newspaper text, dictated messages and spoken text, with up to 26% relative F-score over a
hidden-event language model. This text-only CRF is the closest precedent for RECAP's CRF.
*RECAP differs:* no parser or language-model features, the course's 8-class and joint 12-class label
sets (labels derived from SponSpeech), and evaluation on leak-free splits of spontaneous speech.

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
([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), H1). An exploratory factored-vs-joint study points
the other way on casing: separate models win on the joint metric (+2.08) and on masked casing (+1.49), and
joint training helps only punctuation inside the utterance (−0.83 for the factored pipeline) (validation,
50k sample; §14.5).

## IWSLT TED-talk benchmarks

**Che et al. (2016).** Xiaoyin Che, Cheng Wang, Haojin Yang, Christoph Meinel. *Punctuation
Prediction for Unsegmented Transcript Based on Word Vector.* LREC 2016, pp. 654–658.
<https://aclanthology.org/L16-1103/>
A DNN or CNN over pre-trained word vectors predicts punctuation on TED talks from the IWSLT data.
*RECAP differs:* spontaneous podcast speech instead of prepared talks, no pretrained embeddings in the
course models (the external-reference neural taggers of [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §15 aside),
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
*RECAP differs:* CPU-only classical course models with no pretrained weights (the external-reference neural taggers of [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §15 aside), on
informal speech, with the
utterance-final shortcut reported separately.

**Alam, Khan and Alam (2020).** Tanvirul Alam, Akib Khan, Firoj Alam. *Punctuation Restoration
using Transformer Models for High-and Low-Resource Languages.* W-NUT 2020, pp. 132–142,
doi:10.18653/v1/2020.wnut-1.18. <https://aclanthology.org/2020.wnut-1.18>
Transformer models with data augmentation for English and Bangla punctuation restoration.
*RECAP differs:* no pretrained models in the course work (the external-reference neural taggers of [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §15 aside); it adds the course's joint punctuation and casing task (mode C, with labels we derived from SponSpeech) and a leakage audit.

## Team methods

The team's additions build on these ideas, not on these exact models.

**Lipton, Elkan and Naryanaswamy (2014).** Zachary C. Lipton, Charles Elkan, Balakrishnan Naryanaswamy.
*Optimal Thresholding of Classifiers to Maximize F1 Measure.* ECML PKDD 2014, Lecture Notes in Computer
Science 8725, pp. 225–239, doi:10.1007/978-3-662-44851-9_15.
<https://link.springer.com/chapter/10.1007/978-3-662-44851-9_15> (preprint: <https://arxiv.org/abs/1402.1892>)
Relates a classifier's best achievable F1 to the decision threshold that reaches it; for calibrated
probabilities the F1-optimal threshold is half the optimal F1.
*RECAP differs:* the threshold is applied per token to CRF marginals (1 − P(O) > τ), tuned on validation
for macro F1, and frozen before test.

**Saerens, Latinne and Decaestecker (2002).** Marco Saerens, Patrice Latinne, Christine Decaestecker.
*Adjusting the Outputs of a Classifier to New A Priori Probabilities: A Simple Procedure.* Neural
Computation 14(1), pp. 21–41, doi:10.1162/089976602753284446.
<https://api.crossref.org/works/10.1162/089976602753284446>
Shows how to correct a classifier's posterior probabilities when the class priors change.
*RECAP differs:* prior-adjusted threshold decoding divides each non-O class's CRF marginal by its training
prior raised to γ, with γ chosen on validation (exploratory; [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §14.3).

**Menon et al. (2021).** Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas
Veit, Sanjiv Kumar. *Long-tail learning via logit adjustment.* ICLR 2021. <https://arxiv.org/abs/2007.07314>
Adjusts logits by label frequency, after training or in the loss, so that rare labels are predicted more
often in imbalanced classification.
*RECAP differs:* the same post-hoc idea applied per token to a CRF's marginals, after a separate τ gate
for "any mark", in sequence labelling.

**Lin and Wang (2020).** Binghuai Lin, Liyuan Wang. *Joint Prediction of Punctuation and Disfluency in
Speech Transcripts.* Interspeech 2020, pp. 716–720, doi:10.21437/Interspeech.2020-1277.
<https://www.isca-archive.org/interspeech_2020/lin20c_interspeech.html>
A BERT-based multi-task model with attention between the tasks predicts punctuation and disfluencies
jointly, beating separate models on IWSLT and Switchboard.
*RECAP differs:* a single filler-word and repetition cue in a CRF instead of a disfluency task; on
validation it left macro F1 unchanged and raised comma F1 slightly (+0.21 in B, +0.25 in C).

**Nguyen et al. (2019).** Binh Nguyen, Vu Bao Hung Nguyen, Hien Nguyen, Pham Ngoc Phuong, The-Loc Nguyen,
Quoc Truong Do, Luong Chi Mai. *Fast and Accurate Capitalization and Punctuation for Automatic Speech
Recognition Using Transformer and Chunk Merging.* 2019 22nd Conference of the Oriental COCOSDA
(O-COCOSDA), pp. 1–5, doi:10.1109/O-COCOSDA46868.2019.9041202. <https://arxiv.org/abs/1908.02404>
One Transformer restores capitalization and punctuation in long, unsegmented ASR text by predicting
overlapping chunks and merging them.
*RECAP differs:* each SponSpeech utterance is labelled whole, without sentence segmentation or chunking,
and the utterance-final shortcut that this creates is measured separately (`non_final`).

## Summary: where RECAP sits

RECAP's models are not new. HMM, maximum-entropy and CRF taggers for punctuation (Huang and Zweig 2002;
Liu et al. 2005; Lu and Ng 2010; Ueffing et al. 2013), hidden-event language models (Stolcke and Shriberg
1996; Stolcke et al. 1998) and joint restoration of punctuation and capitalization (Batista et al. 2008;
Gravano et al. 2009; Pahuja et al. 2017) are established, and recent benchmark work uses pretrained
transformers on TED talks with three marks. RECAP applies the course's classical models and its 8- and
12-class modes, with labels derived from SponSpeech, to spontaneous-speech transcripts. Its contributions
are in the data and the evaluation: leak-free splits with the measured cost of leakage, the end-of-utterance
shortcut analysis, and threshold decoding and a filler cue tested on top of the baselines. Exploratory
comparison methods, added after the first test report (EVALUATION_RESULTS_CRF.md §13–14), place the CRF
among its neighbours: a gap classifier without transitions, a Stolcke-style hidden-event LM,
prior-adjusted threshold decoding (Saerens et al. 2002; Menon et al. 2021), question-lookback features, a
factored casing pipeline, and simulated ASR noise.
