"""Generate and finalize query candidates for the Kazakh OOD benchmark.

Two modes:

    # 1. Generate candidates from passages via Gemini
    GEMINI_API_KEY=... python scripts/queries.py generate \
        --passages data/passages.jsonl \
        --per-doc 7 \
        --out data/candidates.csv

    # 2. After manual validation in Google Sheets, finalize to JSONL
    python scripts/queries.py finalize \
        --reviewed data/candidates_reviewed.csv \
        --queries-out data/queries.jsonl \
        --qrels-out data/qrels.jsonl

Three query types per passage:
  - factoid:     direct question using passage keywords
  - paraphrase:  same intent, different words
  - low_overlap: descriptive query avoiding passage keywords (synonyms/paraphrase)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROMPT = """You are generating evaluation queries for a Kazakh information retrieval benchmark.

GIVEN A PASSAGE IN KAZAKH, generate 3 search queries that would retrieve this exact passage:

1. FACTOID — a direct question using key words from the passage.
2. PARAPHRASE — same intent as factoid, but worded differently (different lexicon, same meaning).
3. LOW_OVERLAP — a descriptive query about the passage's content that AVOIDS its key keywords. Use synonyms, paraphrases, or descriptions.

For each query, also provide EVIDENCE — a short quote (3–12 words) from the passage that contains the answer. Evidence MUST be a literal substring from the passage.

OUTPUT STRICT JSON, nothing else:
{
  "factoid":     {"query": "...", "evidence": "..."},
  "paraphrase":  {"query": "...", "evidence": "..."},
  "low_overlap": {"query": "...", "evidence": "..."}
}

RULES:
- All queries MUST be in Kazakh, Cyrillic script. NO Russian, NO English.
- Queries 4–12 words.
- Evidence MUST appear verbatim in the passage.
- If a query type is not feasible for this passage, output null for that key.

PASSAGE:
\"\"\"
{passage}
\"\"\"

JSON:"""


# ---------- sampling ----------

def doc_key(passage_id: str) -> str:
    """Group key from passage id like 'akorda_001_p03' -> 'akorda_001'."""
    parts = passage_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else passage_id


def stratified_sample(passages: list[dict], per_doc: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in passages:
        by_doc[doc_key(p["id"])].append(p)
    sampled: list[dict] = []
    for doc_id in sorted(by_doc):
        items = by_doc[doc_id]
        n = min(per_doc, len(items))
        sampled.extend(rng.sample(items, n))
    return sampled


# ---------- gemini ----------

def call_gemini(passage_text: str, model, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(PROMPT.format(passage=passage_text))
            text = (resp.text or "").strip()
            # Strip code fences if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except Exception as e:
            print(f"  retry {attempt + 1}/{max_retries}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def validate_candidate(candidate: dict | None, passage_text: str) -> tuple[str, str, bool]:
    """Return (query, evidence, evidence_in_passage)."""
    if not candidate or not isinstance(candidate, dict):
        return "", "", False
    query = (candidate.get("query") or "").strip()
    evidence = (candidate.get("evidence") or "").strip()
    in_text = bool(evidence) and evidence in passage_text
    return query, evidence, in_text


# ---------- generate ----------

CSV_COLUMNS = [
    "passage_id", "passage_url", "passage_text",
    "q_factoid", "ev_factoid", "ev_factoid_ok",
    "q_paraphrase", "ev_paraphrase", "ev_paraphrase_ok",
    "q_low_overlap", "ev_low_overlap", "ev_low_overlap_ok",
    "accept_factoid", "accept_paraphrase", "accept_low_overlap",
    "notes",
]


def cmd_generate(args: argparse.Namespace) -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY env var", file=sys.stderr)
        return 1
    try:
        import google.generativeai as genai
    except ImportError:
        print("pip install google-generativeai", file=sys.stderr)
        return 1

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    passages = [json.loads(line) for line in args.passages.read_text(encoding="utf-8").splitlines() if line.strip()]
    sampled = stratified_sample(passages, args.per_doc, args.seed)
    print(f"Loaded {len(passages)} passages, sampled {len(sampled)} ({args.per_doc}/doc)", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, p in enumerate(sampled, 1):
            print(f"[{i}/{len(sampled)}] {p['id']}", file=sys.stderr)
            result = call_gemini(p["text"], model)
            if not result:
                print(f"  WARN: generation failed", file=sys.stderr)
                result = {}

            row: dict = {
                "passage_id": p["id"],
                "passage_url": p.get("url", ""),
                "passage_text": p["text"],
                "accept_factoid": "",
                "accept_paraphrase": "",
                "accept_low_overlap": "",
                "notes": "",
            }
            for qtype in ("factoid", "paraphrase", "low_overlap"):
                q, ev, ok = validate_candidate(result.get(qtype), p["text"])
                row[f"q_{qtype}"] = q
                row[f"ev_{qtype}"] = ev
                row[f"ev_{qtype}_ok"] = "TRUE" if ok else ("FALSE" if q else "")
            writer.writerow(row)
            time.sleep(args.delay)

    print(f"\nDone. {len(sampled)} passages → {args.out}", file=sys.stderr)
    print(f"Open in Google Sheets, fill accept_* columns (TRUE/FALSE), then run 'finalize'.", file=sys.stderr)
    return 0


# ---------- ingest (auto-validate a Gemini TSV against the corpus) ----------

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    for a, b in (("«", '"'), ("»", '"'), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("’", "'")):
        s = s.replace(a, b)
    return s


def _content_stems(s: str) -> list[str]:
    words = re.findall(r"[а-яёәғқңөұүһіА-ЯЁӘҒҚҢӨҰҮҺІ]+", s.lower())
    return [w[:5] for w in words if len(w) >= 3]


def _overlap(query: str, passage: str) -> float:
    qs = set(_content_stems(query))
    if not qs:
        return 0.0
    ps = set(_content_stems(passage))
    return sum(1 for w in qs if w in ps) / len(qs)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Auto-validate a Gemini TSV against the corpus: accept a (passage, type)
    query iff its evidence is a verbatim substring of the real corpus passage."""
    corpus: dict[str, str] = {}
    for line in args.passages.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["id"]] = d["text"]

    rows = []
    with args.tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for r in reader:
            if len(r) >= 8 and r[0] != "passage_id":  # drop embedded header repeats
                rows.append(r)

    types = ("factoid", "paraphrase", "low_overlap")
    queries_out, qrels_out, review_out = [], [], []
    counter = 1
    accepted = defaultdict(int)
    flagged = 0

    for r in rows:
        pid = r[0]
        if pid not in corpus:
            continue
        real = _norm(corpus[pid])
        for i, qtype in enumerate(types):
            query = _norm(r[2 + i * 2])
            evidence = _norm(r[3 + i * 2])
            if not query or query in {"-", ""}:
                continue
            ev_ok = bool(evidence) and evidence in real
            if ev_ok:
                qid = f"ood_q{counter:04d}"
                counter += 1
                queries_out.append({
                    "query_id": qid, "query": query, "type": qtype,
                    "source": "akorda", "overlap": round(_overlap(query, real), 3),
                })
                qrels_out.append({"query_id": qid, "passage_id": pid, "relevance": 1})
                accepted[qtype] += 1
            else:
                flagged += 1
                review_out.append({
                    "passage_id": pid, "type": qtype, "query": query,
                    "evidence": evidence, "reason": "evidence_not_in_corpus_passage",
                })

    args.queries_out.parent.mkdir(parents=True, exist_ok=True)
    with args.queries_out.open("w", encoding="utf-8") as f:
        for q in queries_out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with args.qrels_out.open("w", encoding="utf-8") as f:
        for r in qrels_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if review_out:
        with args.review_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["passage_id", "type", "query", "evidence", "reason"])
            w.writeheader()
            w.writerows(review_out)

    total = sum(accepted.values())
    print(f"Passages in TSV: {len(rows)} | auto-accepted queries: {total} | flagged for review: {flagged}", file=sys.stderr)
    for qtype in types:
        print(f"  {qtype}: {accepted[qtype]}", file=sys.stderr)
    print(f"\n  queries → {args.queries_out}", file=sys.stderr)
    print(f"  qrels   → {args.qrels_out}", file=sys.stderr)
    if review_out:
        print(f"  review  → {args.review_out}  ({flagged} flagged; Gemini drifted from passage)", file=sys.stderr)
    return 0


# ---------- finalize ----------

def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "+", "ok"}


def cmd_finalize(args: argparse.Namespace) -> int:
    queries_out = []
    qrels_out = []
    counter = 1
    type_counts: dict[str, int] = defaultdict(int)

    with args.reviewed.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for qtype in ("factoid", "paraphrase", "low_overlap"):
                if not is_true(row.get(f"accept_{qtype}", "")):
                    continue
                query = (row.get(f"q_{qtype}") or "").strip()
                if not query:
                    continue
                qid = f"ood_q{counter:04d}"
                counter += 1
                queries_out.append({
                    "query_id": qid,
                    "query": query,
                    "type": qtype,
                    "source": "akorda",
                })
                qrels_out.append({
                    "query_id": qid,
                    "passage_id": row["passage_id"],
                    "relevance": 1,
                })
                type_counts[qtype] += 1

    args.queries_out.parent.mkdir(parents=True, exist_ok=True)
    with args.queries_out.open("w", encoding="utf-8") as f:
        for q in queries_out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with args.qrels_out.open("w", encoding="utf-8") as f:
        for r in qrels_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(queries_out)} queries → {args.queries_out}", file=sys.stderr)
    print(f"Wrote {len(qrels_out)} qrels   → {args.qrels_out}", file=sys.stderr)
    for qtype, n in type_counts.items():
        print(f"  {qtype}: {n}", file=sys.stderr)
    return 0


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Sample passages and generate candidate queries via Gemini")
    gen.add_argument("--passages", type=Path, required=True)
    gen.add_argument("--per-doc", type=int, default=7, help="Passages to sample per document")
    gen.add_argument("--out", type=Path, default=Path("data/candidates.csv"))
    gen.add_argument("--model", default="gemini-2.0-flash")
    gen.add_argument("--delay", type=float, default=4.5, help="Seconds between API calls (free tier ~15 RPM)")
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=cmd_generate)

    ing = sub.add_parser("ingest", help="Auto-validate a Gemini TSV against the corpus (evidence-verbatim check)")
    ing.add_argument("--tsv", type=Path, required=True, help="Gemini tab-separated output")
    ing.add_argument("--passages", type=Path, required=True, help="Original passages.jsonl corpus")
    ing.add_argument("--queries-out", type=Path, default=Path("data/queries.jsonl"))
    ing.add_argument("--qrels-out", type=Path, default=Path("data/qrels.jsonl"))
    ing.add_argument("--review-out", type=Path, default=Path("data/review_flagged.csv"))
    ing.set_defaults(func=cmd_ingest)

    fin = sub.add_parser("finalize", help="Convert validated CSV to queries.jsonl + qrels.jsonl")
    fin.add_argument("--reviewed", type=Path, required=True)
    fin.add_argument("--queries-out", type=Path, default=Path("data/queries.jsonl"))
    fin.add_argument("--qrels-out", type=Path, default=Path("data/qrels.jsonl"))
    fin.set_defaults(func=cmd_finalize)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
