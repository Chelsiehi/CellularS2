from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from openai import OpenAI


CORE_PATH = Path(__file__).with_name("stage3-specfusion-specanchor-tfqa.py")
CORE_SPEC = importlib.util.spec_from_file_location("stage3_specanchor_core", CORE_PATH)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise ImportError(f"Cannot load SpecAnchor implementation from {CORE_PATH}")
core = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core)

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_RRC_stage3-CCQA.json",
]
CHUNK_FILES = core.CHUNK_FILES
RAG_FILES = core.RAG_FILES
COMPONENT_DIRS = core.COMPONENT_DIRS
OUTPUT_FILE = Path("stage3-specfusion-specanchor-ccqa.json")
TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY = 8, 0.7, 1, 20

API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""
JUDGE_API_KEY = ""
JUDGE_BASE_URL = ""
JUDGE_MODEL = ""


def generate_answer(client: OpenAI, model: str, question: str, contexts: list[dict], domain: str) -> str:
    prompt = (
        f"{core.domain_prompt(domain)}\n"
        "Answer the cross-clause question by combining all relevant cited evidence chunks.\n\n"
        f"Retrieved Evidence:\n{core.format_contexts(contexts)}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    return core.call_llm(
        client,
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=768,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 CCQA with SpecAnchor hybrid RAG")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output-file", default=str(OUTPUT_FILE))
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME.")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model:
        raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL.")
    for path in DATASETS + list(RAG_FILES.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    generator = OpenAI(api_key=args.api_key, base_url=args.base_url)
    judge = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    indices = {}
    for domain in CHUNK_FILES:
        chunks = core.load_component_space(domain)
        indices[domain] = {
            "chunks": chunks,
            "bm25": core.build_bm25(chunks),
            "dense": core.build_dense(chunks),
        }

    alpha = min(1.0, max(0.0, args.hybrid_alpha))
    results = []
    for path in DATASETS:
        data = json.loads(path.read_text(encoding="utf-8"))
        domain = core.detect_domain(path)
        index = indices[domain]
        for position, item in enumerate(data, 1):
            if position % max(1, args.log_every) == 0 or position == len(data):
                print(f"[CCQA] {path.name} {position}/{len(data)}", flush=True)
            question = item["question"]
            reference = item["answer"]
            gold_evidence = item.get("evidence", "")
            contexts = core.hybrid_retrieve(
                question,
                index["chunks"],
                index["bm25"],
                index["dense"],
                max(1, args.top_k_context),
                alpha,
            )
            answer = generate_answer(generator, args.model_name, question, contexts, domain)
            evidence = core.build_evidence(contexts, domain)
            scores = [
                core.judge_answer(
                    judge,
                    args.judge_model,
                    question,
                    reference,
                    answer,
                    domain,
                )
                for _ in range(max(1, args.grading_passes))
            ]
            results.append(
                {
                    "id": item.get("id"),
                    "question": question,
                    "reference": reference,
                    "gold_evidence": gold_evidence,
                    "model_answer": answer,
                    "evidence": evidence,
                    "score": max(set(scores), key=scores.count),
                    "evidence_correct": core.evidence_in_top_k(contexts, gold_evidence),
                }
            )

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": "CCQA", "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
