from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from openai import OpenAI


def load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


security = load_module("stage3_security_generator", "stage3-specfusion-securityreasoning.py")
anchor = load_module("stage3_specanchor_core", "stage3-specfusion-specanchor-tfqa.py")

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_RRC_stage3-SecurityReasoning.json",
]
OUTPUT_FILE = Path("stage3-specfusion-specanchor-securityreasoning.json")
TOP_K_CONTEXT, HYBRID_ALPHA, LOG_EVERY = 8, 0.7, 20

API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""


def run() -> None:
    parser = argparse.ArgumentParser(description="Stage3 SecurityReasoning with SpecAnchor hybrid RAG")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-file", default=str(OUTPUT_FILE))
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()
    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set API_KEY/BASE_URL/MODEL_NAME.")
    for path in DATASETS + list(anchor.RAG_FILES.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    indices = {}
    for domain in anchor.CHUNK_FILES:
        chunks = anchor.load_component_space(domain)
        indices[domain] = {
            "chunks": chunks,
            "bm25": anchor.build_bm25(chunks),
            "dense": anchor.build_dense(chunks),
        }

    alpha = min(1.0, max(0.0, args.hybrid_alpha))
    results = []
    for path in DATASETS:
        data = json.loads(path.read_text(encoding="utf-8"))
        domain = anchor.detect_domain(path)
        index = indices[domain]
        for position, item in enumerate(data, 1):
            if position % max(1, args.log_every) == 0 or position == len(data):
                print(f"[Security] {path.name} {position}/{len(data)}", flush=True)
            statement = item.get("input", "")
            gold_evidence = item.get("evidence", "")
            contexts = anchor.hybrid_retrieve(
                statement,
                index["chunks"],
                index["bm25"],
                index["dense"],
                max(1, args.top_k_context),
                alpha,
            )
            answer, _ = security.generate_reasoning(
                client,
                args.model_name,
                statement,
                anchor.format_contexts(contexts),
                domain,
            )
            evidence = anchor.build_evidence(contexts, domain)
            results.append(
                {
                    "dataset": path.name,
                    "id": item.get("id"),
                    "input": statement,
                    "gold_evidence": gold_evidence,
                    "model_answer": answer,
                    "evidence": evidence,
                    "evidence_correct": anchor.evidence_in_top_k(contexts, gold_evidence),
                }
            )

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": "SecurityReasoning", "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
