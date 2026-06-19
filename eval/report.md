# Evaluation Report - Grounded Medication-Info RAG

Generation model: `gemini-2.5-flash`  |  judge model: `gemini-3.1-flash-lite`  |  retrieval k = 8  |  refusal threshold = 0.4  |  questions = 24

## Headline metrics

- **Refusal accuracy:** 83% (20/24)
- **False answers (SAFETY-CRITICAL, target 0):** **0**  ✅
- **False refusals:** 4  -> ans-04, ans-08, ans-10, ans-12
- **Faithfulness (answers fully grounded):** 89% (8/9 judged)  -> ungrounded: ans-09

## Retrieval hit rate (answerable questions): k=5 vs k=8

Two granularities. **Drug-level** = the right drug's label appears at all. **Section-level** = the *specific section the question asks about* (e.g. Boxed Warning, Adverse Reactions) appears - this is what actually lets the LLM answer.

| metric | k=5 | k=8 |
|--------|-----|-----|
| drug-level hit rate | 13/13 (100%) | 13/13 (100%) |
| section-level hit rate | 6/13 (46%) | 7/13 (54%) |

- Drug-level is **saturated** (already 100% at k=5), so it can't justify k by itself; k=5->k=8 adds 0 drug hit(s).
- Section-level is the real signal: k=5->k=8 recovers **1** more section hit(s).
- Section still missed at k=8: **ans-01, ans-04, ans-08, ans-10, ans-11, ans-12**.
- **All 4 false refusals (ans-04, ans-08, ans-10, ans-12) are section misses**: the drug is retrieved but the asked-for section isn't, so Layer 2 correctly declines to answer from the wrong section.
- 2 section miss(es) (ans-01, ans-11) still answered - from an adjacent section or a related drug's chunk - so a section miss is the leading *risk factor* for a false refusal, not a strict predictor.
- **Section-level recall is the top thing to improve next** (section-aware retrieval or reranking).

## Score separation by category

Why the two layers exist: the threshold cleanly filters off-topic questions, but drug-like out-of-corpus questions score *above* it, so Layer 2 (the LLM's grounded self-refusal) is required.

| category | n | min | mean | max |
|----------|---|-----|------|-----|
| in_corpus | 13 | 0.505 | 0.630 | 0.758 |
| offtopic | 5 | 0.083 | 0.131 | 0.154 |
| not_in_corpus | 6 | 0.356 | 0.486 | 0.616 |

Threshold = 0.4. Off-topic max should sit below it (caught by Layer 1); not_in_corpus often sits above it (passes Layer 1, caught by Layer 2).

## Caveat

- Generator (`gemini-2.5-flash`) and judge (`gemini-3.1-flash-lite`) are **both from the Gemini family** (split across models only because of free-tier daily quotas). They are not independent, so the faithfulness judge is a known weak check; a stronger/independent judge would be more trustworthy.
