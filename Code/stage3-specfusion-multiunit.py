from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI


CORE_PATH = Path(__file__).with_name("stage3-specfusion-tfqa.py")
CORE_SPEC = importlib.util.spec_from_file_location("stage3_bm25_core", CORE_PATH)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise ImportError(f"Cannot load BM25 implementation from {CORE_PATH}")
core = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core)

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-Multi-Unit.json",
    DATA_ROOT / "5G_NAS_stage3-Multi-Unit.json",
    DATA_ROOT / "5G_RRC_stage3-Multi-Unit.json",
]
CHUNK_FILES = core.CHUNK_FILES
OUTPUT_FILE = Path("stage3-specfusion-multiunit.json")
TOP_K_CONTEXT, LOG_EVERY = 8, 20

API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""

ALLOWED_MULTI = [
    "Integrity Bypass",
    "Ciphering Bypass / Null Encryption",
    "Identity Exposure / Privacy Leakage",
    "Identification Request Surface",
    "Fake Reject / Unauthorized Deregistration / DoS",
    "Replay / Message Injection",
    "Signalling DoS / Resource Exhaustion",
    "Rate-Limit Missing",
    "Anomaly Detection Missing",
    "Emergency-Exception Abuse",
    "Downgrade Attack",
    "Operator Policy / Subscription Exposure",
    "Identity Procedure Abuse",
    "Ciphering Bypass",
    "DoS",
    "Error-Code Oracle",
    "IMSI/IMEI Exposure",
    "Implementation-Dependent Behavior",
    "Linkability / Tracking",
    "Plaintext Identifier Leakage",
    "Plaintext Identifier/Message Exposure",
    "Privacy / Tracking",
    "Privacy/Tracking",
    "Replay Protection Underspecified",
    "Resource Exhaustion",
    "Spoofing / False Paging",
    "Timer Manipulation / Backoff Abuse",
    "Unauthenticated Message Acceptance",
    "Unauthenticated RRC Message",
    "Unencrypted Broadcast",
]


def normalize_label(label: str) -> str:
    value = str(label).strip()
    if value in ALLOWED_MULTI:
        return value
    aliases = {
        "fake reject / dos": "Fake Reject / Unauthorized Deregistration / DoS",
        "identity exposure": "Identity Exposure / Privacy Leakage",
        "replay attack": "Replay / Message Injection",
        "downgrade": "Downgrade Attack",
    }
    return aliases.get(value.lower(), value)


def classify(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    domain: str,
) -> Tuple[str, List[str]]:
    prompt = (
        f"{core.domain_prompt(domain)}\n"
        "You are a cellular specification security analyst. Use the retrieved evidence to classify the input.\n"
        "Return ONLY JSON with keys label_binary and label_multi.\n"
        "Binary label: Vulnerable or non-Vulnerable.\n"
        "Multi-label values must come from this list:\n"
        + "\n".join(f"- {label}" for label in ALLOWED_MULTI)
        + f"\n\nRetrieved Evidence:\n{context}\n\nInput:\n{question}"
    )
    raw = core.call_llm(
        client,
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        obj = json.loads(cleaned)
    except Exception:
        return "", []
    binary = str(obj.get("label_binary", "")).strip()
    values = obj.get("label_multi", [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = []
    multi = [normalize_label(value) for value in values if normalize_label(value)]
    return binary, multi


def run() -> None:
    parser = argparse.ArgumentParser(description="Stage3 Multi-Unit with BM25 chunks")
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
                print(f"[Multi-Unit] {path.name} {position}/{len(data)}", flush=True)
            question = item.get("input", "")
            contexts = core.retrieve(
                question,
                index["chunks"],
                index["bm25"],
                max(1, args.top_k_context),
            )
            binary, multi = classify(
                client,
                args.model_name,
                question,
                core.format_contexts(contexts),
                domain,
            )
            results.append(
                {
                    "dataset": path.name,
                    "id": item.get("id"),
                    "question": question,
                    "reference_binary": item.get("label_binary", ""),
                    "reference_multi": item.get("label_multi", []) or [],
                    "model_label_binary": binary,
                    "model_label_multi": multi,
                }
            )

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": "Multi-Unit", "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
