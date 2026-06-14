"""Scrape Kazakh speeches from akorda.kz and nazarbayev.kz into passages.jsonl.

Usage:
    python scripts/scrape.py --urls urls.txt --out data/passages.jsonl
    python scripts/scrape.py --test   # offline parser test on data/samples/

urls.txt format: one URL per line, # for comments. Domain is auto-detected.

Output JSONL schema (one line per passage):
    {"id": "akorda_001_p03", "source": "akorda", "url": "...",
     "title": "...", "date": "...", "passage_idx": 3, "text": "..."}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install dependencies: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)


UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
TARGET_WORDS = 120
OVERLAP_WORDS = 20


# ---------- parsers ----------

def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_nazarbayev(html: str) -> dict | None:
    """Drupal 8 layout. Title in h1.title-1, body in article.node--type-speeches."""
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one("h1.title-1 .field--name-title") or soup.select_one("h1.title-1")
    title = _clean(title_el.get_text()) if title_el else None

    article = soup.select_one("article.node--type-speeches")
    if not article:
        return None

    body = article.select_one(".field--name-body")
    if not body:
        return None
    # Nested duplicate of same class — take innermost
    nested = body.select(".field--name-body")
    if nested:
        body = nested[-1]

    paragraphs = []
    for p in body.find_all("p"):
        t = _clean(p.get_text())
        if t:
            paragraphs.append(t)
    text = "\n".join(paragraphs)

    date = None
    crumbs = soup.select(".nav-breadcrumbs li")
    if len(crumbs) >= 2:
        date = _clean(crumbs[1].get_text())

    return {"title": title, "text": text, "date": date}


def parse_akorda(html: str) -> dict | None:
    """Custom layout. Title in h2 inside main_block, body in <article>."""
    soup = BeautifulSoup(html, "html.parser")

    main = soup.select_one("#main_block")
    if not main:
        return None

    title_el = main.select_one("h2")
    title = _clean(title_el.get_text()) if title_el else None

    article = main.select_one("article")
    if not article:
        return None

    paragraphs = []
    for p in article.find_all("p"):
        t = _clean(p.get_text())
        if t:
            paragraphs.append(t)
    text = "\n".join(paragraphs)

    # akorda doesn't expose date in article HTML — leave None.
    return {"title": title, "text": text, "date": None}


PARSERS = {
    "akorda.kz": ("akorda", parse_akorda),
    "nazarbayev.kz": ("nazarbayev", parse_nazarbayev),
}


def parse_html(url: str, html: str) -> tuple[str, dict] | None:
    host = urlparse(url).netloc.lower().lstrip("www.")
    for domain, (source, parser) in PARSERS.items():
        if host.endswith(domain):
            parsed = parser(html)
            if parsed and parsed.get("text"):
                return source, parsed
            return None
    raise ValueError(f"No parser for host: {host}")


# ---------- chunking ----------

def chunk_text(text: str, target: int = TARGET_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into ~target-word chunks with overlap. Respects paragraph boundaries softly."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + target, n)
        chunks.append(" ".join(words[i:end]))
        if end == n:
            break
        i = end - overlap
    return chunks


# ---------- pipeline ----------

def fetch(url: str, session: requests.Session, retries: int = 2) -> str:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed {url}: {last_exc}")


def load_urls(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def make_passage_id(source: str, doc_idx: int, passage_idx: int) -> str:
    return f"{source}_{doc_idx:03d}_p{passage_idx:02d}"


def run_scrape(urls_path: Path, out_path: Path, delay: float = 1.0) -> None:
    urls = load_urls(urls_path)
    print(f"Loaded {len(urls)} URLs from {urls_path}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    doc_counters: dict[str, int] = {}
    total_passages = 0

    with out_path.open("w", encoding="utf-8") as out:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}", file=sys.stderr)
            try:
                html = fetch(url, session)
                result = parse_html(url, html)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                continue
            if not result:
                print(f"  WARN: empty parse result", file=sys.stderr)
                continue

            source, parsed = result
            doc_counters[source] = doc_counters.get(source, 0) + 1
            doc_idx = doc_counters[source]
            chunks = chunk_text(parsed["text"])
            print(f"  {source}: {len(parsed['text'].split())} words → {len(chunks)} passages", file=sys.stderr)

            for p_idx, chunk in enumerate(chunks):
                record = {
                    "id": make_passage_id(source, doc_idx, p_idx),
                    "source": source,
                    "url": url,
                    "title": parsed["title"],
                    "date": parsed["date"],
                    "passage_idx": p_idx,
                    "text": chunk,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_passages += 1

            if delay:
                time.sleep(delay)

    print(f"\nDone. {total_passages} passages → {out_path}", file=sys.stderr)
    for source, n in doc_counters.items():
        print(f"  {source}: {n} documents", file=sys.stderr)


# ---------- offline tests ----------

def run_test() -> int:
    samples = [
        ("data/samples/nazarbayev_sample.html", "nazarbayev"),
        ("data/samples/akorda_sample.html", "akorda"),
    ]
    failures = 0
    for path, expected_source in samples:
        p = Path(path)
        if not p.exists():
            print(f"FAIL: missing sample {path}")
            failures += 1
            continue
        html = p.read_text(encoding="utf-8")
        # use parser directly to avoid URL dependency
        _, parser = PARSERS[f"{expected_source}.kz"]
        parsed = parser(html)
        if not parsed or not parsed["text"]:
            print(f"FAIL: {expected_source} parser returned empty")
            failures += 1
            continue
        words = len(parsed["text"].split())
        chunks = chunk_text(parsed["text"])
        print(f"OK  {expected_source}: title={parsed['title'][:60]!r}")
        print(f"       date={parsed['date']!r}  words={words}  chunks={len(chunks)}")
        print(f"       first chunk preview: {chunks[0][:120]!r}")
    if failures:
        print(f"\n{failures} failure(s).")
        return 1
    print("\nAll parsers OK.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", type=Path, help="Path to urls.txt (one URL per line)")
    ap.add_argument("--out", type=Path, default=Path("data/passages.jsonl"))
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")
    ap.add_argument("--test", action="store_true", help="Run offline parser test on data/samples/")
    args = ap.parse_args()

    if args.test:
        return run_test()

    if not args.urls:
        ap.error("--urls is required unless --test")
    if not args.urls.exists():
        ap.error(f"URLs file not found: {args.urls}")

    run_scrape(args.urls, args.out, delay=args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
