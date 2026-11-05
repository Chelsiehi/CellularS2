from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from openai import OpenAI


BASE_PATH = Path(__file__).with_name("stage3-specfusion-specanchor-tfqa.py")
BASE_SPEC = importlib.util.spec_from_file_location("stage3_specanchor_base", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Cannot load SpecAnchor implementation from {BASE_PATH}")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-TFQA.json",
    DATA_ROOT / "5G_NAS_stage3-TFQA.json",
    DATA_ROOT / "5G_RRC_stage3-TFQA.json",
]
KG_DIRS = {
    "4G_NAS": Path("SpecReasoning/4G_NAS_KG_JSON"),
    "5G_NAS": Path("SpecReasoning/5G_NAS_KG_JSON"),
    "5G_RRC": Path("SpecReasoning/5G_RRC_KG_JSON"),
}
OUTPUT_FILE = Path("stage3-specfusion-specanchor-kg-tfqa.json")
TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY = 8, 0.7, 1, 20

API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""
JUDGE_API_KEY = ""
JUDGE_BASE_URL = ""
JUDGE_MODEL = ""


def load_kg(directory: Path) -> list[dict]:
    chunks, seen = [], set()
    if not directory.exists():
        return chunks
    for path in sorted(directory.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        clause_id = str(obj.get("clause_id", "")).strip()
        title = str(obj.get("title", "")).strip()
        label = f"Clause {clause_id}" if clause_id else ""
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        index_text = " ".join(value for value in (label, title, text) if value)
        if not index_text or index_text in seen:
            continue
        seen.add(index_text)
        chunks.append(
            {
                "component_type": "kg",
                "citation_label": label,
                "clause_id": clause_id,
                "title": title,
                "text": text,
                "index_text": index_text,
            }
        )
    return chunks


def load_unified_space(domain: str) -> list[dict]:
    chunks = base.load_component_space(domain) + load_kg(KG_DIRS[domain])
    seen, unique = set(), []
    for chunk in chunks:
        key = (chunk.get("component_type", ""), chunk.get("index_text", ""))
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def select_evidence(contexts: list[dict]) -> dict | None:
    valid = [chunk for chunk in contexts if str(chunk.get("citation_label", "")).strip()]
    for chunk in valid:
        if chunk.get("component_type") == "kg":
            return chunk
    return valid[0] if valid else None


def build_evidence(contexts: list[dict], domain: str) -> str:
    chunk = select_evidence(contexts)
    if chunk is None:
        return ""
    label = str(chunk.get("citation_label", "")).strip()
    title = str(chunk.get("title", "")).strip()
    return f"{base.evidence_prefix(domain)}, {label}" + (f", {title}" if title else "")


def generate_answer(client: OpenAI, model: str, question: str, contexts: list[dict], domain: str) -> str:
    prompt = (
        f"{base.domain_prompt(domain)}\n"
        "Answer the figure or table question using the globally ranked specification and KG evidence.\n\n"
        f"Retrieved Evidence:\n{base.format_contexts(contexts)}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    return base.call_llm(
        client,
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=768,
    )


def run(qa_type: str, datasets: list[Path], output_file: Path, task_instruction: str) -> None:
    parser = argparse.ArgumentParser(description=f"Stage3 {qa_type} with SpecAnchor KG hybrid RAG")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output-file", default=str(output_file))
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME.")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model:
        raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL.")
    for path in datasets + list(base.RAG_FILES.values()) + list(KG_DIRS.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    generator = OpenAI(api_key=args.api_key, base_url=args.base_url)
    judge = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    indices = {}
    for domain in base.CHUNK_FILES:
        chunks = load_unified_space(domain)
        indices[domain] = {
            "chunks": chunks,
            "bm25": base.build_bm25(chunks),
            "dense": base.build_dense(chunks),
        }

    alpha = min(1.0, max(0.0, args.hybrid_alpha))
    results = []
    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        domain = base.detect_domain(path)
        index = indices[domain]
        for position, item in enumerate(data, 1):
            if position % max(1, args.log_every) == 0 or position == len(data):
                print(f"[{qa_type}] {path.name} {position}/{len(data)}", flush=True)
            question = item["question"]
            reference = item["answer"]
            gold_evidence = item.get("evidence", "")
            contexts = base.hybrid_retrieve(
                question,
                index["chunks"],
                index["bm25"],
                index["dense"],
                max(1, args.top_k_context),
                alpha,
            )
            prompt = (
                f"{base.domain_prompt(domain)}\n{task_instruction}\n\n"
                f"Retrieved Evidence:\n{base.format_contexts(contexts)}\n\n"
                f"Question:\n{question}\n\nAnswer:"
            )
            answer = base.call_llm(
                generator,
                args.model_name,
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=768,
            )
            evidence = build_evidence(contexts, domain)
            scores = [
                base.judge_answer(judge, args.judge_model, question, reference, answer, domain)
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
                    "evidence_correct": base.evidence_in_top_k(contexts, gold_evidence),
                }
            )

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": qa_type, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(
        "TFQA",
        DATASETS,
        OUTPUT_FILE,
        "Answer the figure or table question using the globally ranked specification and KG evidence.",
    )
