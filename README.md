# CellularS2-Bench

🌐 **Project Website:** Visit the [CellularS2-Bench project website](https://chelsiehi.github.io/CellularS2/) for an overview of the benchmark, its staged evaluation framework, datasets, and key findings.

This is the official repository for the paper titled **“CellularS2-Bench: A Staged, Evidence-Grounded Benchmark for Cellular Network Security Reasoning”** (NeurIPS 2026 Evaluations and Datasets Track).

## Dataset Structure

CellularS2-Bench is organized into **three progressive stages**, each adding a harder reasoning requirement:

### Stage 1 —  Intra-Clause Specification Comprehension (4,500 + 1,000 questions)

Each question is answerable from a **single specification clause**. No cross-clause integration is required. For each of the three specifications (4G NAS, 5G NAS, and 5G RRC), each of EQA, AQA, and MCQA contains 500 questions.

| Task | Format | Description |
|------|--------|-------------|
| Extractive QA (EQA) |Open-ended answering | Answer must appear verbatim in the clause |
| Abstractive QA (AQA) | Open-ended answering | Answer synthesizes nearby normative text |
| Multi-choice QA (MCQA) | Multiple-choice answering | Select the specification-compliant option |
| TeleQnA (verified) | Multiple-choice answering | Curated from standards subsets |
| Tele-Eval (verified) | Open-ended answering | Curated from 3GPP-origin questions |

### Stage 2 — Evidence-Grounded Answering (4,500 + 500 questions)

Same task types as Stage 1, but every prediction must include **explicit evidence** — clause numbers, table identifiers, or figure labels — that supports the answer. For each of the three specifications, each of EQA-E, AQA-E, and MCQA-E contains 500 questions.

| Task | Format | Description |
|------|--------|-------------|
| EQA-E | Open-ended answering + evidence | Extractive answer with citation |
| AQA-E | Open-ended answering + evidence | Abstractive answer with citation |
| MCQA-E | Multiple-choice answering + evidence | Option selection with citation |
| TSpec-LLM (verified) | Multiple-choice answering + evidence | Reproduced and verified subset |

### Stage 3 —  Vulnerability/Inconsistency Labeling with Evidence and Explanations (1,269 + 2,082 questions)

The hardest stage. Requires **integrating multiple clauses**, reasoning over tables and figures, and performing security-aware interpretation grounded in **43 real-world vulnerabilities**.

| Task | Format | Description |
|------|--------|-------------|
| Cross-Clause QA (CCQA) | Open-ended answering + evidence | Answer requires combining multiple clauses |
| Table & Figure QA (TFQA) | Open-ended answering + evidence | Reasoning over structured multimodal content |
| Vulnerability Labeling | Vulnerability/inconsistency Labeling | Label + vulnerability category + evidence |
| Evidence & Explanation | Vulnerability/inconsistency Labeling | Full attack story + supporting clause set |
| 5GSC (verified) | Vulnerability/inconsistency Labeling | Security vs. non-security sentence classification |
| ConTester (verified) | Vulnerability/inconsistency Labeling | Semantic equivalence over 4G NAS pairs |
| CellularLint (verified) | Vulnerability/inconsistency Labeling | Re-annotated under standard NLI definitions |
| TeleQuAD-Tabular (verified) | Open-ended answering + evidence | QA over specification tables |

---

## Dataset Statistics

| Stage | Newly constructed | Externally verified/corrected | Total |
|-------|------------------:|------------------------------:|------:|
| Stage 1 | 4,500<br>(3 specifications × (500 EQA + 500 AQA + 500 MCQA)) | 1,000<br>(500 TeleQnA + 500 Tele-Eval) | 5,500 |
| Stage 2 | 4,500<br>(3 specifications × (500 EQA-E + 500 AQA-E + 500 MCQA-E)) | 500<br>(500 reconstructed TSpec-LLM) | 5,000 |
| Stage 3 | 1,269<br>(222 CCQA + 747 TFQA + 150 vulnerability-labeling + 150 evidence-and-explanation) | 2,082<br>(1,454 5GSC + 320 ConTester + 55 CellularLint + 253 TeleQuAD-Tabular) | 3,351 |
| **Total** | **10,269 (74.14%)** | **3,582 (25.86%)** | **13,851** |

---

## Source Specifications

| Specification | Version | Description |
|--------------|---------|-------------|
| TS 24.501 | 17.7.1 (Release 17) | 5G NAS — registration, authentication, security mode control |
| TS 38.331 | 17.0.0 (Release 17) | 5G RRC — radio connection, security activation |
| TS 24.301 | 17.6.0 (Release 17) | 4G NAS — EPS procedures, downgrade/compatibility behaviors |

---

# CellularS2-ARI

## SpecFusion (Specification Fusion)

`SpecFusion/` corresponds to the **first stage** of the benchmark.

This stage evaluates a model’s ability to:

- Adapt to the structure and terminology of 3GPP specifications
- Perform clause-level and section-level comprehension
- Understand normative language and procedural descriptions

SpecFusion focuses on **foundational specification understanding**, without requiring external retrieval.

## SpecAnchor (Specification Anchor)

`SpecAnchor/` corresponds to the **second stage** of the benchmark.

This stage evaluates a model’s ability to:

- Ground answers in specific evidence
- Perform retrieval-augmented generation over long and structured documents

SpecAnchor emphasizes **evidence-grounded answering** across specifications.

## SpecReasoning (3GPP Specifications Knowledge Integration Reasoning)

`SpecReasoning/` corresponds to the **third and most advanced stage** of the benchmark.

This stage evaluates a model’s ability to:

- Integrate knowledge from multiple 3GPP specifications
- Reason across clauses, procedures, and protocol layers
- Conduct **security analysis** and identify potential design issues in 3GPP standards

SpecReasoning targets **expert-level reasoning**, where models must synthesize distributed specification knowledge into coherent security conclusions.

