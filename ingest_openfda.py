#!/usr/bin/env python3
"""
ingest_openfda.py - Step 1 of the grounded medication-info RAG.

Pulls drug labels from the public openFDA API, extracts the clinically
relevant sections, and writes a normalized, citable corpus to data/corpus.json.

Design choices (so you can defend each one in an interview):

- CURATED list, not a random pull. A relatable corpus (common meds people
  actually take) makes the demo and the eval questions meaningful, and keeps
  the corpus small enough to embed locally for free.

- We keep each label SECTION as its own unit (indications, dosing, warnings,
  interactions, contraindications, boxed warning). The section is the natural
  *citable grounding unit*: an answer can point to "Warfarin -> Drug Interactions"
  instead of a vague document. Grounding is the backbone of the safety story.

- Every record carries PROVENANCE: the SPL set_id plus a real DailyMed URL, so
  the RAG cites a source a human can actually open and verify.

- PUBLIC, de-identified reference data only. No PHI, no scraping, no copyright.
  This is a deliberate choice and itself a responsible-AI talking point.

Run:
    pip install requests
    python ingest_openfda.py            # live pull from openFDA
    python ingest_openfda.py --selftest # offline parser check, no network
"""

import json
import os
import re
import sys
import time
import urllib.parse

import requests

# --- Config -----------------------------------------------------------------

API = "https://api.fda.gov/drug/label.json"

# ~50 common generics. Chosen for relatability + good interaction/warning
# coverage (warfarin, apixaban, clopidogrel, the SSRIs, etc.) so eval questions
# land on content people recognize.
DRUGS = [
    "warfarin", "apixaban", "rivaroxaban", "clopidogrel", "aspirin",
    "metformin", "glipizide", "sitagliptin", "insulin glargine",
    "atorvastatin", "simvastatin", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "carvedilol", "hydrochlorothiazide", "furosemide",
    "spironolactone", "digoxin", "amoxicillin", "azithromycin",
    "ciprofloxacin", "doxycycline", "cephalexin", "ibuprofen", "naproxen",
    "acetaminophen", "tramadol", "gabapentin", "prednisone", "omeprazole",
    "pantoprazole", "sertraline", "escitalopram", "citalopram", "fluoxetine",
    "duloxetine", "venlafaxine", "alprazolam", "clonazepam", "lorazepam",
    "levothyroxine", "montelukast", "albuterol", "amitriptyline",
    "bupropion", "trazodone", "atenolol", "pravastatin",
]

# openFDA top-level field  ->  human-readable section name.
# We try each; whichever the label actually has, we keep.
SECTION_FIELDS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Cautions",
    "warnings": "Warnings",
    "drug_interactions": "Drug Interactions",
    "adverse_reactions": "Adverse Reactions",
}

# Cap any single section so one giant label can't dominate the corpus.
# We chunk properly in the next step; this is just a sanity bound.
MAX_SECTION_CHARS = 6000


# --- Pure parsing logic (unit-testable, no network) -------------------------

def _join(field_value):
    """openFDA section fields are arrays of strings. Join + tidy."""
    if not field_value:
        return ""
    if isinstance(field_value, list):
        text = "\n".join(str(x) for x in field_value)
    else:
        text = str(field_value)
    return " ".join(text.split()).strip()


def extract_record(label, query_name):
    """
    Turn one raw openFDA label dict into our normalized record:
    {drug, brand, generic, manufacturer, set_id, source_url, sections:[...]}
    Returns None if the label has no usable clinical sections.
    """
    openfda = label.get("openfda", {}) or {}

    def first(key):
        vals = openfda.get(key)
        return vals[0] if isinstance(vals, list) and vals else None

    set_id = label.get("set_id") or first("spl_set_id")
    source_url = (
        f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
        if set_id else "https://open.fda.gov/apis/drug/label/"
    )

    sections = []
    for field, nice_name in SECTION_FIELDS.items():
        text = _join(label.get(field))
        if not text:
            continue
        if len(text) > MAX_SECTION_CHARS:
            text = text[:MAX_SECTION_CHARS].rsplit(" ", 1)[0] + " ..."
        sections.append({"section": nice_name, "text": text})

    if not sections:
        return None

    generic = first("generic_name") or query_name
    return {
        "drug": (first("brand_name") or generic or query_name).title(),
        "generic": (generic or query_name).lower(),
        "brand": first("brand_name"),
        "manufacturer": first("manufacturer_name"),
        "set_id": set_id,
        "source_url": source_url,
        "sections": sections,
    }


# --- Network fetch ----------------------------------------------------------

# Connectors that separate ingredients inside a generic_name string. Combo
# labels often pack several ingredients into ONE string, e.g.
# "sitagliptin and metformin hydrochloride" or
# "olmesartan medoxomil-hydrochlorothiazide", so we must split on these rather
# than trust the list length. (No mono drug in our curated list uses a hyphen
# inside a single ingredient, so splitting on "-" is safe here.)
_INGREDIENT_SEP = re.compile(r"\s+and\s+|[/,;+-]", re.IGNORECASE)


def _ingredients(label):
    """Flatten openfda.generic_name into individual ingredient tokens."""
    openfda = label.get("openfda", {}) or {}
    g = openfda.get("generic_name")
    items = g if isinstance(g, list) else ([g] if g else [])
    out = []
    for s in items:
        for part in _INGREDIENT_SEP.split(str(s).lower()):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _is_mono(label, query_name):
    """True only if the label has exactly ONE ingredient and it is the query."""
    ings = _ingredients(label)
    return len(ings) == 1 and query_name.lower() in ings[0]


def select_label(candidates, query_name):
    """
    Pick the best label among up to 5 candidates.

    We PREFER a mono-ingredient label whose generic_name is exactly one entry
    matching the queried drug. Reason: a single-drug question ("metformin")
    must ground in the metformin-only label, not a combination product
    (e.g. sitagliptin/metformin) whose extra ingredient pollutes retrieval and
    answers a different question. Only if no mono label exists do we fall back
    to any label mentioning the drug, then to the closest (first) result.
    """
    for label in candidates:                       # 1. mono-ingredient match
        if _is_mono(label, query_name):
            return label
    q = query_name.lower()
    for label in candidates:                       # 2. any label naming the drug
        if any(q in g for g in _ingredients(label)):
            return label
    return candidates[0] if candidates else None    # 3. closest available


def fetch_label(query_name, session):
    """
    Query openFDA for one drug. Try generic_name, then brand_name, pulling up
    to 5 results per field so select_label can prefer the mono-ingredient label.
    Returns the chosen raw label dict or None.
    """
    candidates = []
    for field in ("generic_name", "brand_name"):
        search = f'openfda.{field}:"{query_name}"'
        url = f"{API}?search={urllib.parse.quote(search)}&limit=5"
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:   # openFDA returns 404 for "no matches"
                continue
            r.raise_for_status()
            candidates.extend(r.json().get("results", []))
            # Short-circuit as soon as we have a clean mono-ingredient match.
            best = select_label(candidates, query_name)
            if best is not None and _is_mono(best, query_name):
                return best
        except requests.RequestException as e:
            print(f"   ! request error for {query_name} ({field}): {e}")
    return select_label(candidates, query_name)


def run_live():
    os.makedirs("data", exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "rag-medinfo-demo/0.1"})

    corpus, missed, skipped = [], [], []
    seen_setids, seen_generics = set(), set()   # dedupe guards
    for i, name in enumerate(DRUGS, 1):
        print(f"[{i:>2}/{len(DRUGS)}] {name} ...", end=" ")
        label = fetch_label(name, session)
        rec = extract_record(label, name) if label else None
        if not rec:
            missed.append(name)
            print("no usable label")
        elif rec["set_id"] in seen_setids or rec["generic"] in seen_generics:
            # Same SPL document / same generic already captured -> skip the
            # duplicate so it can't appear twice in the index.
            skipped.append(name)
            print("duplicate (skipped)")
        else:
            seen_setids.add(rec["set_id"])
            seen_generics.add(rec["generic"])
            corpus.append(rec)
            print(f"ok ({len(rec['sections'])} sections)")
        time.sleep(0.3)   # be polite; openFDA allows ~240/min unauthenticated

    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    total_sections = sum(len(r["sections"]) for r in corpus)
    print("\n--- corpus locked -------------------------------------------")
    print(f"drugs captured : {len(corpus)}/{len(DRUGS)}")
    print(f"total sections : {total_sections}")
    if skipped:
        print(f"skipped (dup)  : {', '.join(skipped)}")
    if missed:
        print(f"missed         : {', '.join(missed)}")
    print("written        : data/corpus.json")
    # quick field-coverage readout (useful to defend corpus quality)
    cov = {}
    for r in corpus:
        for s in r["sections"]:
            cov[s["section"]] = cov.get(s["section"], 0) + 1
    print("section coverage:")
    for k, v in sorted(cov.items(), key=lambda x: -x[1]):
        print(f"   {v:>3}  {k}")


# --- Offline self-test ------------------------------------------------------

def selftest():
    mock = {
        "set_id": "abc-123",
        "openfda": {
            "brand_name": ["Coumadin"],
            "generic_name": ["WARFARIN SODIUM"],
            "manufacturer_name": ["Example Pharma"],
        },
        "boxed_warning": ["WARNING: BLEEDING RISK. Warfarin can cause major bleeding."],
        "indications_and_usage": ["Warfarin is indicated for the prophylaxis of thrombosis."],
        "drug_interactions": ["Avoid concurrent use with NSAIDs.", "Aspirin increases bleeding risk."],
        "warnings": [],            # empty -> should be skipped
        "adverse_reactions": ["Bleeding, nausea."],
    }
    rec = extract_record(mock, "warfarin")
    assert rec is not None, "record should not be None"
    assert rec["generic"] == "warfarin sodium", rec["generic"]
    assert rec["drug"] == "Coumadin", rec["drug"]
    assert rec["source_url"].endswith("setid=abc-123"), rec["source_url"]
    names = [s["section"] for s in rec["sections"]]
    assert "Warnings" not in names, "empty section must be dropped"
    assert "Drug Interactions" in names
    assert "Boxed Warning" in names
    di = next(s for s in rec["sections"] if s["section"] == "Drug Interactions")
    assert "NSAIDs" in di["text"] and "Aspirin" in di["text"]
    print("selftest passed:", len(rec["sections"]), "sections ->", names)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_live()
