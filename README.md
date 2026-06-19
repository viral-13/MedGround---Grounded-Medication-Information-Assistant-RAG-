# MedGround — Grounded Medication-Information Assistant (RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![RAG](https://img.shields.io/badge/Architecture-RAG-2E75B6)
![Stack](https://img.shields.io/badge/Stack-FAISS%20%C2%B7%20Gemini%20%C2%B7%20Streamlit-444)
![Cost](https://img.shields.io/badge/Cost-%240%20%2F%20local-2E7D32)

> A retrieval-augmented (RAG) assistant that answers questions about common medications using **only** official FDA drug-label text, **cites every claim** to its source, and **refuses to answer** when it can't ground a response — built for trustworthiness in a regulated domain.

A normal chatbot answers from memory and will confidently invent a drug dose. MedGround can only speak from cited sources, and is designed to say *“I don’t have that”* rather than bluff. **In healthcare, a confident wrong answer is worse than an honest refusal** — so the entire system is optimized for verifiability and safe failure over fluency and coverage.

---

## Highlights

- **Grounded answers with real citations** — every response links to the official [DailyMed](https://dailymed.nlm.nih.gov/) label page each claim came from.
- **Two-layer refusal guardrail** — a fast retrieval-similarity gate *plus* the model’s grounded self-refusal, so it declines rather than hallucinates.
- **Measured, not just demoed** — a 24-question evaluation harness reporting refusal accuracy, faithfulness, and retrieval quality, with an honest, diagnosed failure mode.
- **$0 and local** — local embeddings (no embedding API), a free-tier LLM, and public, de-identified data only (**no PHI**).

---

## Demo

> _Add a screen recording / GIF of the Streamlit UI here._

Two example interactions:

- **“Can I take warfarin with aspirin?”** → a grounded answer with four DailyMed citations.
- **“What is the recommended dose of Ozempic?”** → **refused** — Ozempic isn’t in the corpus, so the system declines rather than guess.

---

## Evaluation results

Measured on a fixed 24-question gold set (13 answerable, 5 off-topic, 6 real-but-out-of-corpus drugs):

| Metric | Result |
|---|---|
| **Unsafe (out-of-scope) answers** | **0** — nothing it should have refused was ever answered |
| Refusal accuracy | 83% (20 / 24) |
| Answer faithfulness (LLM-as-judge) | 89% (8 / 9 answered) |
| Retrieval hit rate (correct drug) | 100% |

**Honest finding:** the right *drug* is retrieved 100% of the time, but the specific *section* a question targets reaches the top-k only ~54% of the time — the cause of the few “false refusals.” Section-aware retrieval / reranking is the diagnosed next improvement. (Bigger `k` was tested and barely helps, so it isn’t the lever.)

Full results, score-separation analysis, and methodology are in [`eval/report.md`](eval/report.md).

---

## How it works

**Build phase (run once):**

```
openFDA labels  →  ingest & clean  →  chunk (~150 words)  →  local embeddings  →  FAISS index
```

**Runtime phase (per question):**

```
question
   │  embed
   ▼
retrieve top-k chunks
   │
   ▼
Layer 1: similarity gate  ── top score < 0.40 ?  ──►  REFUSE (no LLM call)
   │ pass
   ▼
grounded generation (LLM sees ONLY retrieved chunks; answer source-only; cite each claim)
   │
   ▼
Layer 2: grounded self-refusal  ── context insufficient ?  ──►  REFUSE
   │ grounded
   ▼
answer + citations (links to DailyMed)
```

**Why two layers?** The cheap gate catches off-topic questions but lets *drug-like-but-absent* questions through (e.g. “dose of Ozempic” scores above the gate). The model’s grounded self-refusal then catches those. Neither layer alone is sufficient — the eval’s score separation proves it.

### Design philosophy: why RAG, not just a system prompt

Instructions change how a model *behaves*; retrieval changes what it *knows*. A prompt can tell a model to “cite the FDA label,” but it can’t hand it the actual label text or verify it obeyed — so citations may be fabricated and there’s nothing to measure faithfulness against. RAG supplies real source text, produces checkable citations, enforces a hard scope boundary, and makes faithfulness measurable. *Trust-me vs. show-me — and in a regulated domain, only show-me survives an audit.*

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Data | openFDA drug labels | Authoritative, public, free, structured |
| Embeddings | `all-MiniLM-L6-v2` (local, 384-dim) | Free, fast, runs on CPU; text never leaves the machine |
| Vector search | FAISS (flat / exact, cosine) | Exact + instant at this scale; no managed DB needed |
| Generation | Google Gemini (free tier) | Behind a provider-agnostic function (swappable) |
| UI | Streamlit | Thin wrapper over the core `query()` function |
| Secrets | python-dotenv (`.env`) | Key kept out of code and git |

---

## Getting started

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your Gemini API key**

Create a `.env` file in the project root (free key from [Google AI Studio](https://aistudio.google.com/)):

```
GEMINI_API_KEY=your_key_here
```

**3. Build the corpus and index** (one-time)

```bash
python ingest_openfda.py      # pulls + cleans FDA labels -> data/corpus.json
python build_index.py         # chunks, embeds locally, builds the FAISS index
```

**4. Ask questions**

```bash
# Command line
python rag.py --ask "Can I take warfarin with aspirin?"

# Web UI
streamlit run app.py          # opens http://localhost:8501
```

**5. Run the evaluation** (optional)

```bash
python eval/run_eval.py        # writes eval/report.md
```

> **Note:** the free Gemini tier has a small daily request cap (resets midnight Pacific). The harness caches results and is resumable, so a mid-run quota cut-off is harmless. Layer-1 refusals work with no LLM call.

---

## Project structure

```
.
├── ingest_openfda.py       # Step 1: pull + clean FDA labels  -> data/corpus.json
├── build_index.py          # Step 2: chunk + local embeddings + FAISS index
├── rag.py                  # Core RAG: retrieval + two-layer refusal + cited generation
├── app.py                  # Streamlit UI (thin layer over query())
├── eval/
│   ├── run_eval.py         # Evaluation harness (cached, resumable)
│   ├── gold_questions.json # 24-question gold set with answer key
│   ├── results.json        # Per-question results
│   └── report.md           # Evaluation report
├── data/corpus.json        # (generated) cleaned corpus
├── index/                  # (generated) FAISS index + chunk metadata
├── requirements.txt
└── .env                    # GEMINI_API_KEY (gitignored)
```

---

## Limitations & roadmap

- **Section-level retrieval recall** (~54%) is the primary weakness — causes occasional false refusals on basic questions. **Next up:** section-aware retrieval / reranking.
- **“Grounded” ≠ “safely framed”** — an answer can be fully cited yet lead with an edge case. Faithfulness can’t catch framing; a real product needs framing review with a domain expert.
- **Single-family LLM judge** — generator and faithfulness judge share a model family (split only for quota reasons); an independent judge would be stronger.
- **Scope:** 50 drugs by design; adding more is a corpus change, no retraining.

---

## Responsible AI & data

- **Public, de-identified data only — no PHI**, by design.
- **Privacy-aware:** embeddings are computed locally so text isn’t sent out to be vectorized; only non-sensitive public data passes through the free-tier LLM.
- **Auditable:** every answer carries checkable citations to official label pages.
- **Safe failure:** the system refuses rather than guesses, and every answer ends with a not-medical-advice disclaimer.

---

## Disclaimer

This is a **demonstration project**, not an approved or validated clinical tool. It is **not medical advice**. Always consult a qualified healthcare professional and the official prescribing information.

---

## License

MIT — feel free to add a `LICENSE` file if you want to publish under it.
