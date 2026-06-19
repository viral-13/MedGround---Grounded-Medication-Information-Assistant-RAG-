#!/usr/bin/env python3
"""
rag.py - Step 3 of the grounded medication-info RAG.

The core pipeline: retrieve from the FAISS index, then either REFUSE or generate
a strictly-grounded, cited answer with two layers of guardrail.

Two-layer refusal (the safety backbone of this project):

  Layer 1 - retrieval gate. If the best chunk's cosine similarity is below a
  threshold, we REFUSE before spending an LLM call. Off-topic ("capital of
  France") and out-of-corpus ("Ozempic") questions never retrieve anything
  close, so this catches them cheaply and deterministically.

  Layer 2 - grounded generation. The model sees ONLY the retrieved chunks and is
  told to answer from them alone, cite every claim, and self-refuse (can_answer:
  false) when the context is insufficient. So even if retrieval squeaks past the
  gate, the model gets a second chance to refuse rather than hallucinate.

The LLM lives behind a single provider-agnostic generate(prompt) function, so the
backend (currently Gemini free tier) can be swapped without touching the pipeline.

Run:
    pip install -r requirements.txt
    # .env must contain: GEMINI_API_KEY=...
    python rag.py --ask "Can I take warfarin with aspirin?"
"""

import argparse
import json
import os
import re
import sys
import time

from dotenv import load_dotenv

# Reuse the exact same embedding model + index paths as the builder, so query
# vectors live in the same space as the indexed chunks.
from build_index import load_model, embed, INDEX_PATH, CHUNKS_PATH

load_dotenv()

# Drug-label text contains Unicode (e.g. "<=" rendered as the symbol). Force
# UTF-8 stdout so printing answers never dies on Windows' default cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# --- Config -----------------------------------------------------------------

RETRIEVAL_THRESHOLD = 0.40        # Layer-1 gate: refuse below this cosine score
# DEFAULT_K is a recall knob (how many chunks the LLM sees) to be calibrated in
# the Step-5 eval; the refusal gate always keys off the single top-1 score.
DEFAULT_K = 8
GEMINI_MODEL = "gemini-3.1-flash-lite"  # free-tier flash model (see eval/report.md caveat re: model/quota)
DISCLAIMER = "This is reference information from drug labels, not medical advice."
REFUSAL_MSG = ("I don't have grounded information on that in my drug-label "
               "sources, so I can't answer it.")

_RESOURCES = None   # lazily-loaded (index, chunks, model) cache
_CLIENT = None      # lazily-loaded Gemini client cache


# --- Provider-agnostic LLM call ---------------------------------------------

def _client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY not set - put it in a .env file.")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def generate(prompt, max_retries=5):
    """
    Single swappable entry point to the LLM. Returns raw text.
    Retries with exponential backoff on 429 (free-tier rate limits).
    """
    from google.genai import errors

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = _client().models.generate_content(
                model=GEMINI_MODEL, contents=prompt)
            return resp.text or ""
        except errors.APIError as e:
            code = getattr(e, "code", None)
            msg = str(e)
            # Retry transient free-tier conditions: 429 rate limit and 503
            # model-overloaded. Both clear on their own; back off and retry.
            transient = (code in (429, 503) or "429" in msg or "503" in msg
                         or "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg)
            if transient and attempt < max_retries - 1:
                reason = "rate-limited" if (code == 429 or "429" in msg) else "overloaded"
                print(f"   ({reason}, retrying in {delay:.0f}s ...)", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Gemini rate-limited after retries")


# --- Prompt + parsing -------------------------------------------------------

def build_prompt(question, retrieved):
    """Give the model ONLY the retrieved chunks, labeled [Drug - Section]."""
    context = "\n\n".join(
        f"[{r['drug']} - {r['section']}]\n{r['text']}" for r in retrieved)
    return f"""You are a careful medication-information assistant. You answer ONLY \
from the context below, which are excerpts from official FDA drug labels.

Rules:
- Use ONLY the provided context. Do NOT use any outside knowledge.
- Cite the drug and section for every claim, using the exact [Drug - Section] labels.
- If the context does not contain enough information to answer the question, set
  "can_answer" to false rather than guessing.
- The "answer" text MUST end with exactly this sentence: "{DISCLAIMER}"

Context:
{context}

Question: {question}

Respond with STRICT JSON only (no markdown, no code fences), exactly this shape:
{{"can_answer": true or false, "answer": "string", "citations": ["Drug - Section", ...]}}"""


def parse_json(raw):
    """Parse the model's JSON, tolerating code fences / stray prose."""
    if not raw:
        return None
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)   # grab first {...} block
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def map_citations(cite_strings, retrieved):
    """
    Map the model's "Drug - Section" citation strings back to real chunks so we
    can attach a verifiable source_url to each. Falls back to the retrieved set
    if the model's strings don't match, so provenance is never lost.
    """
    out, seen = [], set()
    for cs in cite_strings or []:
        csl = str(cs).lower()
        match = next(
            (r for r in retrieved
             if r["drug"].lower() in csl and r["section"].lower() in csl), None)
        if not match:   # looser: drug-only match
            match = next((r for r in retrieved if r["drug"].lower() in csl), None)
        if match:
            key = (match["drug"], match["section"])
            if key not in seen:
                seen.add(key)
                out.append({"drug": match["drug"], "section": match["section"],
                            "source_url": match["source_url"]})
    if not out:   # never drop provenance - cite the unique retrieved chunks
        for r in retrieved:
            key = (r["drug"], r["section"])
            if key not in seen:
                seen.add(key)
                out.append({"drug": r["drug"], "section": r["section"],
                            "source_url": r["source_url"]})
    return out


# --- Retrieval --------------------------------------------------------------

def _resources():
    global _RESOURCES
    if _RESOURCES is None:
        import faiss
        if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
            sys.exit("Index not found - run `python build_index.py` first.")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
        _RESOURCES = (index, chunks, load_model())
    return _RESOURCES


def retrieve(question, k=DEFAULT_K):
    index, chunks, model = _resources()
    qvec = embed(model, [question])
    scores, idxs = index.search(qvec, k)
    out = []
    for score, i in zip(scores[0], idxs[0]):
        c = chunks[int(i)]
        out.append({"drug": c["drug"], "generic": c["generic"],
                    "section": c["section"], "score": float(score),
                    "text": c["text"], "source_url": c["source_url"]})
    return out


def _result(question, answer, refused, citations, retrieved, top_score):
    return {
        "question": question,
        "answer": answer,
        "refused": refused,
        "citations": citations,
        "retrieved": [{"drug": r["drug"], "generic": r["generic"],
                       "section": r["section"], "score": r["score"],
                       "text": r["text"]} for r in retrieved],
        "top_score": top_score,
    }


# --- The core RAG -----------------------------------------------------------

def query(question, threshold=RETRIEVAL_THRESHOLD, k=DEFAULT_K):
    retrieved = retrieve(question, k)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    # Layer 1: retrieval gate - refuse cheaply, before any LLM call.
    if top_score < threshold:
        return _result(question, REFUSAL_MSG, True, [], retrieved, top_score)

    # Layer 2: grounded generation with self-refusal.
    raw = generate(build_prompt(question, retrieved))
    if os.environ.get("RAG_DEBUG"):
        print(f"   [debug] raw LLM output: {raw!r}", file=sys.stderr)
    data = parse_json(raw)
    if not data or not data.get("can_answer"):
        return _result(question, REFUSAL_MSG, True, [], retrieved, top_score)

    answer = str(data.get("answer", "")).strip()
    if not answer:
        return _result(question, REFUSAL_MSG, True, [], retrieved, top_score)
    if not answer.endswith(DISCLAIMER):   # enforce the disclaimer ourselves
        answer = answer.rstrip() + "\n\n" + DISCLAIMER

    citations = map_citations(data.get("citations"), retrieved)
    return _result(question, answer, False, citations, retrieved, top_score)


# --- CLI --------------------------------------------------------------------

def _print(res):
    print(f"\nQ: {res['question']}")
    print(f"top retrieval score: {res['top_score']:.3f}  "
          f"(threshold {RETRIEVAL_THRESHOLD}) -> "
          f"{'REFUSED' if res['refused'] else 'ANSWERED'}")
    print("-" * 70)
    print(res["answer"])
    if res["citations"]:
        print("\nCitations:")
        for c in res["citations"]:
            print(f"  - {c['drug']} -> {c['section']}\n    {c['source_url']}")
    print(f"\nretrieved (top {len(res['retrieved'])}):")
    for r in res["retrieved"]:
        print(f"  {r['score']:.3f}  {r['drug']} -> {r['section']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grounded medication-info RAG.")
    ap.add_argument("--ask", required=True, help="the question to answer")
    args = ap.parse_args()
    _print(query(args.ask))
