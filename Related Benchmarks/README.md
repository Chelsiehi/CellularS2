# Related Benchmarks

This directory contains the externally sourced datasets that were verified, corrected, or reconstructed for comparison with CellularS2-Bench. Together, these sources contribute 3,582 evaluation instances.

## Included datasets

| Dataset | Repository file | Instances | CellularS2 use |
|---------|-----------------|----------:|----------------|
| TeleQnA | `teleqna.json` | 500 | Stage 1 multiple-choice QA |
| Tele-Eval | `tele_eval.jsonl` | 500 | Stage 1 open-ended QA |
| Reconstructed TSpec-LLM | `gpt4-MCQA.json` | 500 | Stage 2 MCQA with evidence |
| 5GSC / SPEC5G | `5gsc.csv` | 1,454 | Stage 3 security classification |
| ConTester | `constester.txt` | 320 | Stage 3 semantic-relation classification |
| CellularLint | `cellularlint.txt` | 55 | Stage 3 entailment/neutral/contradiction classification |
| TeleQuAD-Tabular | `TeleQuAD.json` | 253 | Stage 3 table QA with evidence |

## Supporting content

- `table/` contains 60 table images referenced by the TeleQuAD-Tabular subset. Filenames include the originating specification, clause, and table identifier.
- `text/` contains RAG-ready document collections derived for retrieval experiments.

## Data provenance

These resources are redistributed or adapted for research comparison. Users should consult the original benchmark publications and licenses before further redistribution or commercial use. Corrections made for CellularS2 evaluation do not replace the authoritative upstream datasets or 3GPP specifications.
