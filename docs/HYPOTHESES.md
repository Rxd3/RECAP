# Team hypotheses

The six hypotheses below were written in the team plan before the CRF results (the plan is not in the repo): the team execution plan (*RECAP — Team Execution Plan v2*,
whose status block is dated Thu Oct 29, 00:50, i.e. 2026-10-29). They are
copied verbatim, H1–H6 exactly as written, from that plan's section 5. Their outcomes, supported or not,
are in [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) (Team hypotheses).

---

## 5. Hypotheses (test them; report the outcome either way)

- **H1:** Joint mode C gets higher PERIOD F1 than punctuation-only mode A (compared on the A projection), because of `PERIOD → FIRST_UPPER` transitions.
- **H2:** Threshold decoding raises recall on rare non-O classes, at some cost in precision.
- **H3:** Question cues raise QUESTION F1 over the basic CRF.
- **H4:** Filler and discourse cues raise COMMA F1 on spontaneous speech.
- **H5:** A large share of PERIOD F1 comes from utterance-final tokens. This is measured by the shortcut baseline and the non-final metric.
- **H6:** CRF beats HMM within each mode. Report the gap with paired CIs.
