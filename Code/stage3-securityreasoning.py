from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_RRC_stage3-SecurityReasoning.json",
]
OUTPUT_FILE = Path("stage3-securityreasoning.json")
LOG_EVERY = 20
MAX_RETRIES = 6
RETRY_BASE_SLEEP = 1.2
RETRY_JITTER = 0.5

# Choose your settings directly here:
# API_KEY = "your_generation_api_key"
# BASE_URL = "https://your-openai-compatible-endpoint/v1"
# MODEL_NAME = "your_generation_model"
API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""


def domain_prompt(path: Path) -> str:
    n = path.name
    if n.startswith("4G_NAS"):
        return "You are a 4G NAS expert (3GPP TS 24.301)."
    if n.startswith("5G_NAS"):
        return "You are a 5G NAS expert (3GPP TS 24.501)."
    if n.startswith("5G_RRC"):
        return "You are a 5G RRC expert (3GPP TS 38.331)."
    return "You are a telecom specification expert."


def call_llm(client: OpenAI, model_name: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(model=model_name, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep((RETRY_BASE_SLEEP * (2 ** i)) + random.random() * RETRY_JITTER)
    return f"[ERROR] {repr(last_err)}"


def normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def generate_reasoning(client: OpenAI, statement: str, path: Path) -> Tuple[str, str]:
    prompt = (
        f"{domain_prompt(path)}\n"
        "Analyze the following specification statement for security implications.\n"
        "Explain whether it represents a vulnerability and provide concise security reasoning.\n"
        "Return ONLY strict JSON with keys answer and evidence.\n"
        "The evidence value must be the most likely 3GPP clause, table, or figure citation.\n\n"
        f"Input:\n{statement}"
    )
    raw = call_llm(
        client,
        MODEL_NAME,
        [{"role": "user", "content": prompt}],
        0.2,
        1536,
    )
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        obj = json.loads(cleaned)
        answer = str(obj.get("answer", "")).strip() or raw
        evidence = str(obj.get("evidence", "")).strip()
        return answer, evidence
    except Exception:
        return raw, ""


def main() -> None:
    global API_KEY, BASE_URL, MODEL_NAME, OUTPUT_FILE, LOG_EVERY
    ap = argparse.ArgumentParser(description="Stage3 SecurityReasoning (release-safe)")
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--output-file", default=str(OUTPUT_FILE))
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args()

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    OUTPUT_FILE = Path(args.output_file)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    if not API_KEY or not BASE_URL or not MODEL_NAME:
        raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME.")
    gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    all_results = []

    for path in DATASETS:
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[Security] {path.name} {i}/{len(data)}", flush=True)
            statement = item.get("input", "")
            gold_evidence = item.get("evidence", "")
            model_answer, evidence = generate_reasoning(gen_client, statement, path)
            all_results.append(
                {
                    "dataset": path.name,
                    "id": item.get("id"),
                    "input": statement,
                    "gold_evidence": gold_evidence,
                    "model_answer": model_answer,
                    "evidence": evidence,
                    "evidence_correct": bool(
                        gold_evidence
                        and normalize_evidence(evidence)
                        == normalize_evidence(gold_evidence)
                    ),
                }
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(
            {"type": "SecurityReasoning", "results": all_results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
