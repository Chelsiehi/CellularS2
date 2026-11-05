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
    DATA_ROOT / "4G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_RRC_stage3-CCQA.json",
]
OUTPUT_FILE = Path("stage3-ccqa.json")

GRADING_PASSES = 1
MAX_RETRIES = 6
RETRY_BASE_SLEEP = 1.2
RETRY_JITTER = 0.5
LOG_EVERY = 20

# Choose your settings directly here:
# API_KEY = "your_generation_api_key"
# BASE_URL = "https://your-openai-compatible-endpoint/v1"
# JUDGE_API_KEY = "your_deepseek_api_key"
# JUDGE_BASE_URL = "https://api.deepseek.com"
# JUDGE_MODEL = "deepseek-chat"
# MODEL_NAME = "your_generation_model"
API_KEY = ""
BASE_URL = ""
JUDGE_API_KEY = ""
JUDGE_BASE_URL = ""
JUDGE_MODEL = ""
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


def generate_answer(client: OpenAI, question: str, path: Path) -> Tuple[str, str]:
    p = (
        f"{domain_prompt(path)}\n"
        "Answer concisely and accurately.\n"
        "Also provide the most likely evidence clause reference.\n"
        "Respond ONLY as JSON with keys 'answer' and 'evidence'.\n\n"
        f"Question:\n{question}"
    )
    raw = call_llm(client, MODEL_NAME, [{"role": "user", "content": p}], 0.2, 512)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        j = json.loads(raw)
        return str(j.get("answer", "")).strip() or raw, str(j.get("evidence", "")).strip()
    except Exception:
        return raw, ""


def judge_answer(client: OpenAI, question: str, reference: str, candidate: str, path: Path) -> Tuple[int, str]:
    p = (
        "Grade this answer against the reference.\n\n"
        "Score 2 = Correct (covers core reasons, no obvious contradiction).\n"
        "Score 1 = Partially Correct (direction right but missing key points or includes unreliable extensions).\n"
        "Score 0 = Incorrect (main reason wrong or conflicts with spec).\n\n"
        f"Question:\n{question}\n\nReference answer:\n{reference}\n\nModel answer:\n{candidate}\n\n"
        "Return ONLY strict JSON with keys: score, justification."
    )
    raw = call_llm(client, JUDGE_MODEL, [{"role": "system", "content": domain_prompt(path)}, {"role": "user", "content": p}], 0.0, 256)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        j = json.loads(raw)
        s = int(j.get("score", 0))
        just = str(j.get("justification", "")).strip()
    except Exception:
        m = re.search(r"\b([0-2])\b", raw)
        s = int(m.group(1)) if m else 0
        just = raw
    return max(0, min(s, 2)), just


def grade_multiple(client: OpenAI, q: str, ref: str, ans: str, path: Path) -> Tuple[int, str]:
    scores = []
    just = ""
    for _ in range(max(1, GRADING_PASSES)):
        s, j = judge_answer(client, q, ref, ans, path)
        scores.append(s)
        just = just or j
    return max(set(scores), key=scores.count), just


def main() -> None:
    global API_KEY, BASE_URL, JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL, MODEL_NAME, OUTPUT_FILE, GRADING_PASSES, LOG_EVERY
    ap = argparse.ArgumentParser(description="Stage3 CCQA (release-safe)")
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    ap.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--output-file", default=str(OUTPUT_FILE))
    ap.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args()

    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME.")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model:
        raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL.")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_FILE = Path(args.output_file)
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    all_results = []

    for path in DATASETS:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[CCQA] {path.name} {i}/{len(data)}", flush=True)
            q = item["question"]
            ref = item["answer"]
            gold_ev = item.get("evidence", "")
            ans, ev = generate_answer(gen_client, q, path)
            s, just = grade_multiple(judge_client, q, ref, ans, path)
            ok_ev = bool(gold_ev and normalize_evidence(ev) == normalize_evidence(gold_ev))
            out.append(
                {
                    "id": item.get("id"),
                    "question": q,
                    "reference": ref,
                    "gold_evidence": gold_ev,
                    "model_answer": ans,
                    "evidence": ev,
                    "score": s,
                    "evidence_correct": ok_ev,
                }
            )

        all_results.extend(out)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps({"type": "CCQA", "results": all_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
