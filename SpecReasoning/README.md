# SpecReasoning

`SpecReasoning/` contains clause-level representations of the three source specifications used for cross-clause reasoning, security analysis, and retrieval augmented with the **Taxonomy-based KG**.

## Coverage

| Protocol domain | Source specification | Clause JSON directory | Taxonomy-based KG directory |
|-----------------|----------------------|-----------------------|---------------------------|
| 4G NAS | TS 24.301 version 17.6.0 (Release 17) | `4G_NAS/` | `4G_NAS_KG_JSON/` |
| 5G NAS | TS 24.501 version 17.7.1 (Release 17) | `5G_NAS_JSON/` | `5G_NAS_KG_JSON/` |
| 5G RRC | TS 38.331 version 17.0.0 (Release 17) | `5G_RRC_JSON/` | `5G_RRC_KG_JSON/` |

The clause and Taxonomy-based KG directories provide separate views of the specifications and do not always have a one-to-one file mapping.

## Clause representation

Each `Clause_*.json` file represents a specification clause or subclause. Its main fields are:

- `spec`: source specification identifier.
- `clause_id`: clause or subclause number.
- `title`: clause title.
- `text`: normalized clause content.
- `has_text`: whether usable textual content is present.
- `tables` and `tables_count`: tables associated with the clause.
- `figures` and `figures_count`: figures associated with the clause.

## Taxonomy-based KG representation

Each `Clause_*_kg.json` file contains:

- `spec`, `clause_id`, and `title`: source metadata.
- `entities`: entities extracted from the clause.
- `relations`: structured relationships among those entities.

The Taxonomy-based KG directories are used by the `*-specanchor-kg*` evaluation scripts. Those scripts retrieve from the paragraph/component space and the Taxonomy-based KG space, globally rerank the combined candidates, and prefer Taxonomy-based KG evidence when an adequate result is available.

## Intended use

These files support Stage 3 tasks that require integrating information across clauses or protocol components. They are machine-readable derivatives of the cited 3GPP specifications and should be interpreted together with their specification and clause metadata.
