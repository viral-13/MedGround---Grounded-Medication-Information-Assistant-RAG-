#!/usr/bin/env python3
"""
build_index.py - Step 2 of the grounded medication-info RAG.

Chunks the section-level corpus, embeds each chunk with a small LOCAL model,
and builds a FAISS cosine-similarity index. Everything runs offline and free
once the model is cached.

Design choices (so you can defend each one later):

- SECTION-AWARE CHUNKING. We never merge across sections or across drugs: a
  chunk must stay inside one (drug, section) so its citation is unambiguous.
  Within a section we greedily pack ~150 words per chunk and carry ~1 sentence
  of overlap into the next chunk, so a fact split across a boundary still has a
  chance of landing whole in some chunk. A short section stays a single chunk.

- PROVENANCE ON EVERY CHUNK. Each chunk copies the drug's set_id + source_url,
  so any retrieved chunk can be cited back to a real DailyMed page. Grounding is
  the whole point of this project.

- L2-NORMALIZED VECTORS + IndexFlatIP. Inner product on unit vectors == cosine
  similarity. Flat (exact, brute-force) search is the right call at this scale
  (a few hundred chunks): it is exact and instant. Approximate indexes
  (IVF / HNSW) only earn their complexity at hundreds of thousands+ of vectors.

Run:
    pip install -r requirements.txt
    python build_index.py                                  # build the index
    python build_index.py --query "side effects of sertraline"   # retrieval smoke test
"""

import argparse
import json
import os
import re
import sys

import numpy as np

# --- Config -----------------------------------------------------------------

CORPUS_PATH = "data/corpus.json"
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, ~80MB on first-run download
TARGET_WORDS = 150                # soft target chunk size
OVERLAP_SENTENCES = 1             # carry ~1 sentence across the boundary


# --- Chunking (pure, no model needed) ---------------------------------------

def split_sentences(text):
    """Lightweight sentence splitter. Good enough for label prose; avoids a
    heavyweight NLP dependency for a $0 local stack."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_section(text):
    """Greedy, sentence-aware packing to ~TARGET_WORDS with 1-sentence overlap.
    Returns a list of chunk strings (>=1). Never crosses section boundaries
    because it is only ever called with one section's text."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    cur, cur_words = [], 0
    for sent in sentences:
        n = len(sent.split())
        # If adding this sentence overflows the target and we already have
        # content, close the chunk and start the next one with overlap.
        if cur and cur_words + n > TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur = cur[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            cur_words = sum(len(s.split()) for s in cur)
        cur.append(sent)
        cur_words += n

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(corpus):
    """Flatten the corpus into chunk records, each carrying full provenance."""
    chunks = []
    for rec in corpus:
        for sec in rec["sections"]:
            for piece in chunk_section(sec["text"]):
                chunks.append({
                    "chunk_id": len(chunks),
                    "drug": rec["drug"],
                    "generic": rec["generic"],
                    "section": sec["section"],
                    "text": piece,
                    "source_url": rec["source_url"],
                    "set_id": rec["set_id"],
                })
    return chunks


# --- Embedding + index ------------------------------------------------------

def load_model():
    # Imported lazily so --selftest-style pure functions don't pay the cost.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed(model, texts):
    """Encode + L2-normalize so inner product == cosine similarity."""
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


def build():
    import faiss

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    chunks = build_chunks(corpus)
    if not chunks:
        sys.exit("No chunks produced - is data/corpus.json populated?")

    print(f"Loaded {len(corpus)} drugs -> {len(chunks)} chunks. "
          f"Embedding with {MODEL_NAME} ...")
    model = load_model()
    vecs = embed(model, [c["text"] for c in chunks])

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # exact cosine search on normalized vectors
    index.add(vecs)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    # chunks.json order MUST match FAISS vector order (row i == chunk i).
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    avg_words = sum(len(c["text"].split()) for c in chunks) / len(chunks)
    print("\n--- index built ---------------------------------------------")
    print(f"total chunks   : {len(chunks)}")
    print(f"avg chunk len  : {avg_words:.1f} words")
    print(f"model          : {MODEL_NAME}")
    print(f"dimension      : {dim}")
    print(f"written        : {INDEX_PATH}, {CHUNKS_PATH}")


# --- Retrieval smoke test ---------------------------------------------------

def query(text, k=5):
    import faiss

    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        sys.exit("Index not found - run `python build_index.py` first.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    model = load_model()
    qvec = embed(model, [text])
    scores, idxs = index.search(qvec, k)

    print(f"\nquery: {text}")
    print("-" * 70)
    for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), 1):
        c = chunks[i]
        snippet = " ".join(c["text"].split()[:30])
        print(f"{rank}. {c['drug']} -> {c['section']}  (cos={score:.3f})")
        print(f"   {snippet} ...")


# --- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build / query the medication RAG index.")
    ap.add_argument("--query", help="run a top-5 retrieval smoke test instead of building")
    args = ap.parse_args()

    if args.query:
        query(args.query)
    else:
        build()
