# CellularS2-Bench Datasets

`Datasets/` contains the newly constructed portion of CellularS2-Bench. The tasks progress from clause comprehension to evidence-grounded answering and cross-component security reasoning.

## Specification coverage

| Dataset prefix | Protocol | Specification |
|----------------|----------|---------------|
| `4G_NAS` | 4G Non-Access Stratum | TS 24.301 version 17.6.0 (Release 17) |
| `5G_NAS` | 5G Non-Access Stratum | TS 24.501 version 17.7.1 (Release 17) |
| `5G_RRC` | 5G Radio Resource Control | TS 38.331 version 17.0.0 (Release 17) |

## Filename convention

Files follow this pattern:

```text
<protocol-domain>_stage<stage>-<task>.json
```

For example, `5G_NAS_stage2-MCQA.json` contains Stage 2 evidence-grounded multiple-choice questions derived from TS 24.501.

## Tasks

| Stage | Task | Purpose | Main fields |
|-------|------|---------|-------------|
| 1 | EQA | Extractive clause-level QA | `id`, `question`, `answer` |
| 1 | AQA | Abstractive clause-level QA | `id`, `type`, `question`, `answer` |
| 1 | MCQA | Multiple-choice clause comprehension | `id`, `type`, `question`, `options`, `answer` |
| 2 | EQA/AQA | QA with explicit supporting evidence | Stage 1 fields plus `evidence` |
| 2 | MCQA | Multiple choice with explicit evidence | Stage 1 MCQA fields plus `evidence` |
| 3 | CCQA | Cross-clause question answering | `id`, `type`, `question`, `answer`, `evidence` |
| 3 | TFQA | Table- and figure-grounded QA | `id`, `type`, `question`, `answer`, `evidence` |
| 3 | Multi-Unit | Binary and multi-label vulnerability classification | `id`, `input`, `label_binary`, `label_multi` |
| 3 | SecurityReasoning | Structured vulnerability or non-vulnerability reasoning | `id`, `input`, `label`, `evidence`, task-specific reasoning fields |

Each Stage 1 and Stage 2 domain/task file contains 500 records. Stage 3 contains 222 CCQA, 747 TFQA, 150 Multi-Unit, and 150 SecurityReasoning records across the three protocol domains.

## JSON format

Each dataset file is a JSON array. Evidence fields may contain clause, table, or figure references. Options and some label fields can be structured values rather than plain strings, so consumers should preserve their JSON types when loading and exporting the data.
