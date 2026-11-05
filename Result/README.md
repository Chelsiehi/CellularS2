# Evaluation Results

`Result/` stores model predictions and evaluation outputs produced under different CellularS2 stages and retrieval settings. It contains results only; benchmark inputs are maintained in `Datasets/` and external benchmark sources in `Related Benchmarks/`.

## Organization

Results under `data/` are grouped first by model and then, where available, by evaluation stage or system configuration.

The evaluated models and their corresponding result-directory prefixes are:

| Result directory prefix | Model | Provider |
|-------------------------|-------|--------------------|
| `gpt5.6terra/` | GPT-5.6 Terra | OpenAI |
| `sonnet5/` | Claude Sonnet 5 | Anthropic  |
| `deepseek_*` | DeepSeek-V3.2-Exp | DeepSeek |
| `qwen_*` | Qwen-plus-2025-12-01 | Alibaba Cloud |
| `llama17b_*` | Llama-4-Scout-17B-16E-Instruct | Meta AI |
| `qwen30b_*` | Qwen3-30B-A3B-Instruct-2507 | Alibaba Cloud |

Common configuration names are:

| Name | Context used at inference time |
|------|--------------------------------|
| `base` | No retrieval context |
| `specfusion` | Retrieved SpecFusion chunks/components |
| `specfusion-specanchor` | Citation-aware hybrid retrieval |
| `specfusion-specanchor-kg` | Hybrid retrieval augmented with the Taxonomy-based KG |


## Result format

Current evaluation scripts write a JSON object with a task label and a list of predictions:

```json
{
  "type": "MCQA",
  "results": [
    {
      "id": "MCQA-0001"
    }
  ]
}
```

The fields inside each result depend on the task:

- Stage 1 free-form QA records contain the question, reference answer, model answer, and correctness score.
- Stage 1 MCQA records additionally contain options and a Boolean correctness value.
- Stage 2 and evidence-grounded Stage 3 records contain gold evidence, retrieved/generated evidence, answer correctness, and evidence correctness as applicable.
- Multi-Unit records contain binary and multi-label references and predictions.
- SecurityReasoning records contain the input, gold evidence, model answer, returned evidence, and evidence-correctness result.

Older imported result files may use legacy field names or omit the top-level `type`. Compare runs only after confirming that their schemas and judging configurations are compatible.
