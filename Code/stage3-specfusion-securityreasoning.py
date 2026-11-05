from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Tuple

from openai import OpenAI


CORE_PATH = Path(__file__).with_name("stage3-specfusion-tfqa.py")
CORE_SPEC = importlib.util.spec_from_file_location("stage3_bm25_core", CORE_PATH)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise ImportError(f"Cannot load BM25 implementation from {CORE_PATH}")
core = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core)

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_RRC_stage3-SecurityReasoning.json",
]
CHUNK_FILES = core.CHUNK_FILES
OUTPUT_FILE = Path("stage3-specfusion-securityreasoning.json")
TOP_K_CONTEXT, LOG_EVERY = 8, 20

API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""


def normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def generate_reasoning(
    client: OpenAI,
    model: str,
    statement: str,
    context: str,
    domain: str,
) -> Tuple[str, str]:
    prompt = (
        f"{core.domain_prompt(domain)}\n"
        "Analyze the specification statement for security implications using the retrieved evidence.\n"
        "Explain whether it represents a vulnerability and provide concise security reasoning.\n"
        "Return ONLY strict JSON with keys answer and evidence.\n"
        "The evidence must be the most likely 3GPP clause, table, or figure citation.\n\n"
        f"Retrieved Evidence:\n{context}\n\nInput:\n{statement}"
    )
    raw = core.call_llm(
        client,
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    try:
        obj = json.loads(re.sub(r"```json|```", "", raw).strip())
        return str(obj.get("answer", "")).strip() or raw, str(obj.get("evidence", "")).strip()
    except Exception:
        return raw, ""


def run() -> None:
    parser = argparse.ArgumentParser(description="Stage3 SecurityReasoning with BM25 chunks")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-file", default=str(OUTPUT_FILE))
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()
    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set API_KEY/BASE_URL/MODEL_NAME.")
    for path in DATASETS + list(CHUNK_FILES.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    indices = {}
    for domain, path in CHUNK_FILES.items():
        chunks = core.load_chunks(path)
        indices[domain] = {"chunks": chunks, "bm25": core.build_bm25(chunks)}

    results = []
    for path in DATASETS:
        data = json.loads(path.read_text(encoding="utf-8"))
        domain = core.detect_domain(path)
        index = indices[domain]
        for position, item in enumerate(data, 1):
            if position % max(1, args.log_every) == 0 or position == len(data):
                print(f"[Security] {path.name} {position}/{len(data)}", flush=True)
            statement = item.get("input", "")
            gold_evidence = item.get("evidence", "")
            contexts = core.retrieve(
                statement,
                index["chunks"],
                index["bm25"],
                max(1, args.top_k_context),
            )
            answer, evidence = generate_reasoning(
                client,
                args.model_name,
                statement,
                core.format_contexts(contexts),
                domain,
            )
            results.append(
                {
                    "dataset": path.name,
                    "id": item.get("id"),
                    "input": statement,
                    "gold_evidence": gold_evidence,
                    "model_answer": answer,
                    "evidence": evidence,
                    "evidence_correct": bool(
                        gold_evidence
                        and normalize_evidence(evidence)
                        == normalize_evidence(gold_evidence)
                    ),
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
