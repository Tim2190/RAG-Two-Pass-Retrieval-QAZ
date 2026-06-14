# Kazakh OOD Confirmatory Benchmark — Data Collection

Workspace for collecting a held-out, out-of-distribution Kazakh corpus from official sources, to be uploaded into the [primary benchmark repository](https://huggingface.co/datasets/Tim2190/kaz-rag-search-benchmark) as a confirmatory test set.

## Purpose

The primary benchmark (Wikipedia, n=300) establishes the headline results. This corpus answers the natural reviewer question: *do the same conclusions hold on text from a different domain and register?* The new corpus comes from presidential and government addresses — formal Kazakh, distinct from encyclopedic style.

## Sources

Both are Kazakh public-domain content with stable per-document URLs:

- **akorda.kz** — Office of the President of Kazakhstan: speeches (`/kz/speeches`)
- **nazarbayev.kz** — N. Nazarbayev Foundation: speeches, interviews, addresses (`/kk/soylegen-sozder-suhbattar-men-zholdaular-26104624`)

## Pipeline

This repo only handles **passage collection**. Query generation and relevance judgments happen separately and land in the benchmark repo.

```
URLs list  →  scrape  →  clean + chunk  →  passages.jsonl  →  (upload to benchmark repo)
```

### Usage

```bash
pip install requests beautifulsoup4

# 1. Verify parsers on offline samples (no network)
python scripts/scrape.py --test

# 2. Edit data/urls.txt — add ~22 speech URLs (one per line)

# 3. Scrape
python scripts/scrape.py --urls data/urls.txt --out data/passages.jsonl

# 4. Upload data/passages.jsonl to the benchmark repository
```

### Output schema

One JSON object per line:

```json
{
  "id": "akorda_001_p03",
  "source": "akorda",
  "url": "https://akorda.kz/kz/...",
  "title": "Мемлекет басшысы ...",
  "date": "20 наурыз 2020",
  "passage_idx": 3,
  "text": "..."
}
```

- `id`: `{source}_{doc_index}_p{passage_index}`
- `date`: extracted where available (nazarbayev.kz has it in breadcrumbs; akorda.kz does not expose it in article HTML, so left `null`)
- Passages are ~120 words with 20-word overlap, same chunking convention as the primary benchmark

## Repo contents

```
scripts/scrape.py    — scraper with per-site parsers, chunking, offline test mode
data/samples/        — two HTML fixtures used by --test (committed for reproducibility)
data/urls.txt        — list of speech URLs to scrape (fill in manually)
data/passages.jsonl  — output (gitignored; lands in the benchmark repo)
```

## Why scraping is run locally

Outbound HTTPS to `akorda.kz` and `nazarbayev.kz` is blocked from cloud Claude environments (HTTP 403). The script is designed to run on your local machine; only the offline `--test` mode runs in any environment.

## Next steps after collection

1. Upload `passages.jsonl` to the [benchmark repo](https://huggingface.co/datasets/Tim2190/kaz-rag-search-benchmark) under a new partition (e.g. `confirmatory/`)
2. Generate query candidates per passage via a free LLM API (Groq Llama 3.3 / Gemini Flash)
3. Native-speaker validation: keep / rewrite / drop (target ≈150 final pairs)
4. Re-run the existing dense/lexical pipelines on the new partition and compare category-level rankings against Wikipedia-n=300 results
