# SpecAnchor

`SpecAnchor/` contains citation-aware retrieval corpora for grounding model outputs in explicit 3GPP evidence.

## Retrieval corpora

The retrieval corpora cover the following protocol domains and specifications:

- 4G NAS — TS 24.301.
- 5G NAS — TS 24.501.
- 5G RRC — TS 38.331.

Each domain directory contains two JSONL files:

- `*_paragraph_rag_chunks.jsonl` provides context-rich retrieval units.
- `*_sentence_rag_chunks.jsonl` provides fine-grained retrieval units.

Every line is an independent JSON object with the following fields:

```json
{
  "clause_id": "5.4.2",
  "clause_title": "...",
  "text": "..."
}
```

## Retrieval role

The `specfusion-specanchor` scripts build a hybrid index over these chunks and the relevant SpecFusion components. Sparse relevance is computed with BM25, while dense relevance is approximated with TF-IDF vectors projected by truncated SVD. The two normalized scores are combined using a configurable weight, and the highest-ranked candidates form the model context.

The `specfusion-specanchor-kg` variants additionally retrieve from the **Taxonomy-based KG** under `SpecReasoning/`, combine those candidates with the paragraph/component candidates, and perform global reranking. Evidence is generated from the Taxonomy-based KG result when suitable KG evidence is available; otherwise, paragraph evidence is used.
