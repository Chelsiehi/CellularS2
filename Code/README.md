# Evaluation Code

`Code/` contains the generation, retrieval, and evaluation entry points for the three CellularS2 stages. Scripts use OpenAI-compatible generation and judge endpoints, with optional local Hugging Face generation in the Stage 1 base implementation.

## Experiment variants

| Suffix | Retrieval behavior |
|--------|--------------------|
| `base` | No retrieval; answers are generated from the question and options only |
| `specfusion` | BM25 retrieval over SpecFusion paragraph/component chunks |
| `specfusion-specanchor` | Hybrid BM25 and TF-IDF/SVD retrieval over citation-aware and multimodal components |
| `specfusion-specanchor-kg` | Joint retrieval and global reranking over paragraph/component candidates and Taxonomy-based KG candidates, with Taxonomy-based KG-preferred evidence generation |

Hybrid scripts use a weighted score of the normalized sparse and dense similarities. The default settings are generally `top_k = 8` and `alpha = 0.7`; both can be changed through the scripts' command-line options where exposed.

## Script groups

### Stage 1

- `stage1-base.py`
- `stage1-specfusion.py`
- `stage1-specfusion-specanchor.py`
- `stage1-specfusion-specanchor-kg.py`

Each script evaluates AQA, EQA, and MCQA inputs.

### Stage 2

- `stage2-base.py`
- `stage2-specfusion.py`
- `stage2-specfusion-specanchor.py`
- `stage2-specfusion-specanchor-kg.py`

Stage 2 evaluates the same three QA formats while also returning and judging evidence.

### Stage 3

Stage 3 uses task-specific entry points for CCQA, TFQA, Multi-Unit classification, and SecurityReasoning. Base scripts are provided for all task families, and filenames containing `specfusion`, `specanchor`, or `kg` identify the corresponding retrieval variant. In these filenames, `kg` specifically denotes the **Taxonomy-based KG** variant.

Some Stage 3 scripts dynamically reuse the TFQA implementation in the same directory. Keep the full set of Stage 3 files together when copying or running them.

## Requirements

The scripts require Python 3 and, depending on the selected variant:

```text
openai
numpy
scikit-learn
torch
transformers
```

`stage2-specfusion.py` additionally imports `llama14b_local_client`, which must be available on `PYTHONPATH` for that specific local/hybrid setup.

## Configuration and execution

Dataset, chunk, RAG, Taxonomy-based KG, and output locations are declared near the top of each script. Paths are resolved relative to the current working directory, so confirm `DATA_ROOT`, `CHUNK_FILES`, `RAG_FILES`, `COMPONENT_DIRS`, and `KG_DIRS` before running an experiment.

Most scripts expose their runtime settings through `--help`. A base Stage 1 invocation from the repository root can use:

```bash
DATA_ROOT=Datasets python Code/stage1-base.py \
  --api-key "$GENERATION_API_KEY" \
  --base-url "$GENERATION_BASE_URL" \
  --model-name "$GENERATION_MODEL" \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-model "$JUDGE_MODEL"
```

Do not commit API keys. Use command-line arguments or environment variables supported by the selected script.

## Outputs

Current scripts write a top-level JSON object containing `type` and `results`. Free-form, multiple-choice, evidence-grounded, Multi-Unit, and SecurityReasoning tasks use different result fields; see `Result/README.md` for the output conventions.
