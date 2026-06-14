# Kazakh OOD Confirmatory Benchmark

Confirmatory out-of-distribution evaluation set for Kazakh retrieval — built to verify that conclusions from the primary [Wikipedia-based benchmark](https://huggingface.co/datasets/Tim2190/kaz-rag-search-benchmark) (n=300) and the native-validated semantic-gap subset (n=127, Sprint 3) generalize to non-encyclopedic Kazakh text.

## Purpose

The Wikipedia benchmark establishes the primary result: lexical retrieval with Kazakh morphological normalization beats naive multilingual dense embeddings, and a hybrid BM25+stemmer ⊕ dense pipeline gives the best measured performance. This dataset is a **confirmatory test**: do the same systems retain their relative ranking on text from a different domain and register?

A held-out, independently sourced benchmark de-risks the published results against a likely reviewer objection: *"do your conclusions depend on Wikipedia-specific style?"*

## Sources

Two Kazakh public-domain corpora with distinct registers:

- **akorda.kz** — official presidential office: addresses, speeches, articles
- **nazarbayev.kz** — N. Nazarbayev Foundation: articles and speeches

Specific sections to be selected manually to avoid press releases and ensure substantive text. Sections list is curated, not bulk-scraped.

## Target

- ~150 native-speaker-validated query–passage pairs (collect ~200–220 candidates, expect 20–25% drop during validation)
- Passages chunked at ~120 words, consistent with the primary benchmark
- Category structure: TBD (to mirror the primary benchmark, or focus on the validated semantic-gap design from Sprint 3)

## Relationship to other repositories

- [`Kaz-RAG-search-benchmark`](https://huggingface.co/datasets/Tim2190/kaz-rag-search-benchmark) — primary benchmark (Wikipedia, n=300)
- Sprint 3 semantic-gap subset (n=127) — native-validated, low lexical overlap
- This repo — confirmatory OOD benchmark (akorda + nazarbayev)
