# SpecFusion

`SpecFusion/` contains the unified multimodal component space extracted from the three 3GPP specifications. Text, tables, and figures are represented separately so that evaluation pipelines can retrieve the component type best suited to a question.

## Source specifications

| Domain | Specification | Version |
|--------|---------------|---------|
| 4G NAS | TS 24.301 | 17.6.0 (Release 17) |
| 5G NAS | TS 24.501 | 17.7.1 (Release 17) |
| 5G RRC | TS 38.331 | 17.0.0 (Release 17) |

## Directory structure

| Domain | Text chunks | Tables | Figures |
|--------|------------:|-------:|--------:|
| 4G NAS | 769 paragraph / 6,605 sentence | 145 | 130 |
| 5G NAS | 777 paragraph / 9,864 sentence | 100 | 248 |
| 5G RRC | 320 paragraph / 5,564 sentence | 3 | 46 |

The nine component directories follow the pattern `<domain>_<component>`:

- `*_Text/` contains paragraph- and sentence-level JSONL files.
- `*_Table/` contains one JSON file per extracted table.
- `*_Figure/` contains one JSON file per extracted figure.

## Data formats

Text files are JSONL: each line is an independent object with a `text` field. Paragraph chunks provide broader context, while sentence chunks support more precise matching.

Table JSON files use `table_title`, `columns`, and `rows`. Figure JSON files use `figure_id`, `title`, `type`, and `content`, where `content` is the textual representation used by downstream retrieval and reasoning code.

## Role in CellularS2

SpecFusion supplies normalized specification components to the `specfusion` evaluation variants. The table and figure representations are also included in the component space used by multimodal and SpecAnchor retrieval, especially for TFQA.
