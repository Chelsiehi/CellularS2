from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

# ===================== Configuration ===================== #

DATA_ROOT = Path("")
DATASETS_AQA = [
    DATA_ROOT / "4G_NAS_stage2-AQA.json",
    DATA_ROOT / "5G_NAS_stage2-AQA.json",
    DATA_ROOT / "5G_RRC_stage2-AQA.json",
]
DATASETS_EQA = [
    DATA_ROOT / "4G_NAS_stage2-EQA.json",
    DATA_ROOT / "5G_NAS_stage2-EQA.json",
    DATA_ROOT / "5G_RRC_stage2-EQA.json",
]
DATASETS_MCQA = [
    DATA_ROOT / "4G_NAS_stage2-MCQA.json",
    DATA_ROOT / "5G_NAS_stage2-MCQA.json",
    DATA_ROOT / "5G_RRC_stage2-MCQA.json",
]

OUTPUT_ROOT = Path("stage2-base")

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


# ===================== Utilities ===================== #


def detect_domain(dataset_path: Path) -> str:
    name = dataset_path.name
    if name.startswith("4G_NAS"):
        return "4G_NAS"
    if name.startswith("5G_NAS"):
        return "5G_NAS"
    if name.startswith("5G_RRC"):
        return "5G_RRC"
    raise ValueError(f"Unknown dataset domain for {dataset_path}")


def domain_prompt(domain: str) -> str:
    if domain == "4G_NAS":
        return "You are a 4G NAS expert (3GPP TS 24.301)."
    if domain == "5G_NAS":
        return "You are a 5G NAS expert (3GPP TS 24.501)."
    if domain == "5G_RRC":
        return "You are a 5G RRC expert (3GPP TS 38.331)."
    return "You are a telecom specification expert."


def call_llm(client: OpenAI, model_name: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            sleep_s = (RETRY_BASE_SLEEP * (2 ** attempt)) + random.random() * RETRY_JITTER
            time.sleep(sleep_s)
    return f"[ERROR] {repr(last_err)}"


def normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ===================== Generation / Judge ===================== #


def generate_freeform_answer(gen_client: OpenAI, question: str, qa_type: str, domain: str) -> Tuple[str, str, List[Dict]]:
    if qa_type == "EQA":
        task_inst = "This is an extractive question. Provide a short direct answer."
    else:
        task_inst = "This is an analytical question. Provide concise and technically accurate reasoning."

    prompt = (
        f"{domain_prompt(domain)}\n"
        f"{task_inst}\n"
        "Also provide the most likely evidence clause and 8 relevant contexts from your knowledge.\n"
        "Return strict JSON: {\"answer\":..., \"evidence\":..., \"retrieved_contexts\": [...]}\n\n"
        f"Question:\n{question}"
    )

    raw = call_llm(gen_client, MODEL_NAME, [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=512)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        obj = json.loads(raw)
        answer = str(obj.get("answer", "")).strip() or raw
        evidence = str(obj.get("evidence", "")).strip()
        retrieved = obj.get("retrieved_contexts", [])
        if not isinstance(retrieved, list):
            retrieved = []
    except Exception:
        answer = raw
        evidence = ""
        retrieved = []
    return answer, evidence, retrieved


def judge_answer(judge_client: OpenAI, question: str, reference: str, candidate: str, domain: str) -> Tuple[int, str]:
    prompt = (
        "Grade this answer against the reference.\n\n"
        "Score 2 = Correct (covers core reasons, no obvious contradiction).\n"
        "Score 1 = Partially Correct (direction right but missing key points or includes unreliable extensions).\n"
        "Score 0 = Incorrect (main reason wrong or conflicts with spec).\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Model answer:\n{candidate}\n\n"
        "Return ONLY strict JSON with keys: score, justification."
    )
    raw = call_llm(
        judge_client,
        JUDGE_MODEL,
        [
            {"role": "system", "content": domain_prompt(domain)},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        obj = json.loads(raw)
        score = int(obj.get("score", 0))
        just = str(obj.get("justification", "")).strip()
    except Exception:
        m = re.search(r"\b([0-2])\b", raw)
        score = int(m.group(1)) if m else 0
        just = raw
    return max(0, min(score, 2)), just


def grade_multiple(judge_client: OpenAI, q: str, ref: str, ans: str, domain: str) -> Tuple[int, str]:
    scores = []
    just = ""
    for _ in range(max(1, GRADING_PASSES)):
        s, j = judge_answer(judge_client, q, ref, ans, domain)
        scores.append(s)
        just = just or j
    return max(set(scores), key=scores.count), just


# ===================== Pipeline ===================== #


def process_freeform(gen_client: OpenAI, judge_client: OpenAI, datasets: List[Path], qa_type: str) -> None:
    results = []
    total_score = 0
    total_n = 0
    evidence_correct = 0
    evidence_total = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)

        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[{qa_type}] {path.name} {i}/{len(data)}", flush=True)
            q = item["question"]
            ref = item["answer"]
            gold_ev = item.get("evidence")

            ans, ev, retrieved = generate_freeform_answer(gen_client, q, qa_type, domain)
            score, just = grade_multiple(judge_client, q, ref, ans, domain)
            total_score += score

            ok_ev = bool(gold_ev and normalize_evidence(ev) == normalize_evidence(gold_ev))
            if gold_ev:
                evidence_total += 1
                evidence_correct += int(ok_ev)

            results.append(
                {
                    "id": item.get("id"),
                    "question": q,
                    "reference": ref,
                    "gold_evidence": gold_ev,
                    "model_answer": ans,
                    "evidence": ev,
                    "score": score,
                    "evidence_correct": ok_ev,
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / f"{qa_type.lower()}.json"
    out_path.write_text(
        json.dumps(
            {
                "type": qa_type,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def process_mcqa(gen_client: OpenAI, datasets: List[Path]) -> None:
    results = []
    total_n = 0
    correct = 0
    evidence_correct = 0
    evidence_total = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)

        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[MCQA] {path.name} {i}/{len(data)}", flush=True)
            q = item["question"]
            opts = item["options"]
            gold = str(item["answer"]).strip().upper()
            gold_ev = item.get("evidence")

            prompt = (
                f"{domain_prompt(domain)}\n"
                "Choose the correct option and reply with A/B/C/D only.\n\n"
                f"Q: {q}\n"
                + "\n".join(f"{k}: {v}" for k, v in opts.items())
            )
            raw = call_llm(gen_client, MODEL_NAME, [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=64)
            m = re.search(r"\b([A-D])\b", raw.upper())
            pred = m.group(1) if m else "X"

            ev_prompt = (
                f"{domain_prompt(domain)}\n"
                "Provide the most likely evidence clause and 8 relevant contexts from your knowledge.\n"
                "Return strict JSON: {\"evidence\":..., \"retrieved_contexts\": [...]}\n\n"
                f"Question:\n{q}"
            )
            ev_raw = call_llm(gen_client, MODEL_NAME, [{"role": "user", "content": ev_prompt}], temperature=0.1, max_tokens=256)
            ev_raw = re.sub(r"```json|```", "", ev_raw).strip()
            try:
                ev_obj = json.loads(ev_raw)
                ev = str(ev_obj.get("evidence", "")).strip()
                retrieved = ev_obj.get("retrieved_contexts", [])
                if not isinstance(retrieved, list):
                    retrieved = []
            except Exception:
                ev = ev_raw
                retrieved = []

            ok = pred == gold
            correct += int(ok)
            ok_ev = bool(gold_ev and normalize_evidence(ev) == normalize_evidence(gold_ev))
            if gold_ev:
                evidence_total += 1
                evidence_correct += int(ok_ev)

            results.append(
                {
                    "id": item.get("id"),
                    "question": q,
                    "options": opts,
                    "gold": gold,
                    "gold_evidence": gold_ev,
                    "model_answer": pred,
                    "evidence": ev,
                    "correct": ok,
                    "evidence_correct": ok_ev,
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / "mcqa.json"
    out_path.write_text(
        json.dumps(
            {
                "type": "MCQA",
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    global API_KEY, BASE_URL, JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL, MODEL_NAME
    global OUTPUT_ROOT, GRADING_PASSES, LOG_EVERY

    parser = argparse.ArgumentParser(description="Stage-2 base generation (release-safe).")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME (or pass args).")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model:
        raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL (or pass args).")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_ROOT = Path(args.output_root)
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS_AQA + DATASETS_EQA + DATASETS_MCQA:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    process_mcqa(gen_client, DATASETS_MCQA)
    process_freeform(gen_client, judge_client, DATASETS_AQA, "AQA")
    process_freeform(gen_client, judge_client, DATASETS_EQA, "EQA")


if __name__ == "__main__":
    main()
