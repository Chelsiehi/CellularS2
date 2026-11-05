from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Tuple

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== Configuration ===================== #

DATA_ROOT = Path(os.getenv("DATA_ROOT", ""))
DATASETS_AQA = [
    DATA_ROOT / "4G_NAS_stage1-AQA.json",
    DATA_ROOT / "5G_NAS_stage1-AQA.json",
    DATA_ROOT / "5G_RRC_stage1-AQA.json",
]
DATASETS_EQA = [
    DATA_ROOT / "4G_NAS_stage1-EQA.json",
    DATA_ROOT / "5G_NAS_stage1-EQA.json",
    DATA_ROOT / "5G_RRC_stage1-EQA.json",
]
DATASETS_MCQA = [
    DATA_ROOT / "4G_NAS_stage1-MCQA.json",
    DATA_ROOT / "5G_NAS_stage1-MCQA.json",
    DATA_ROOT / "5G_RRC_stage1-MCQA.json",
]

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "stage1-base"))

GRADING_PASSES = 1
MAX_NEW_TOKENS = 1024
MAX_RETRIES = 6
RETRY_BASE_SLEEP = 1.2
RETRY_JITTER = 0.5

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


def generate_text_local(model, tokenizer, prompt: str, temperature: float = 0.2) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_ids = output[0][len(inputs.input_ids[0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def generate_text_api(client: OpenAI, prompt: str, temperature: float = 0.2) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            sleep_s = (RETRY_BASE_SLEEP * (2 ** attempt)) + random.random() * RETRY_JITTER
            time.sleep(sleep_s)
    return f"[ERROR] {repr(last_err)}"


# ===================== Answer Generation ===================== #


def generate_answer(model, tokenizer, gen_client: OpenAI | None, question: str, qa_type: str, domain: str) -> str:
    if qa_type == "EQA":
        task_inst = (
            "This is an extractive question.\n"
            "- If possible, quote the exact statement that answers the question.\n"
            "- Otherwise, answer concisely based on your own knowledge."
        )
    elif qa_type == "AQA":
        task_inst = (
            "This is an analytical question.\n"
            "- Reason about conditions, states, and expected behavior."
        )
    else:
        task_inst = ""

    prompt = (
        f"{domain_prompt(domain)}\n\n"
        "Use your own knowledge of the specification to answer the question.\n\n"
        f"{task_inst}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    if gen_client is not None:
        return generate_text_api(gen_client, prompt, temperature=0.2)
    return generate_text_local(model, tokenizer, prompt, temperature=0.2)


def judge_call_with_retry(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            sleep_s = (RETRY_BASE_SLEEP * (2 ** attempt)) + random.random() * RETRY_JITTER
            time.sleep(sleep_s)
    return f"[ERROR] {repr(last_err)}"


def judge_answer(client: OpenAI, question: str, reference: str, candidate: str, domain: str) -> Tuple[int, str]:
    prompt = (
        f"{domain_prompt(domain)}\n\n"
        "Grade this answer.\n"
        "Score 2 = Correct\n"
        "Score 1 = Partially correct\n"
        "Score 0 = Incorrect or contradicts spec\n\n"
        f"Question:\n{question}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Model answer:\n{candidate}\n\n"
        "Respond ONLY as JSON with keys score and justification."
    )
    raw = judge_call_with_retry(client, domain_prompt(domain), prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        j = json.loads(raw)
        score = int(j.get("score", 0))
        justification = str(j.get("justification", "")).strip()
    except Exception:
        score = 0
        justification = raw
    score = max(0, min(score, 2))
    return score, justification


def grade_multiple(client: OpenAI, q: str, ref: str, ans: str, domain: str) -> Tuple[int, str]:
    scores = []
    justification = ""
    for _ in range(max(1, GRADING_PASSES)):
        s, j = judge_answer(client, q, ref, ans, domain)
        scores.append(s)
        justification = justification or j
    return max(set(scores), key=scores.count), justification


# ===================== Freeform QA (AQA / EQA) ===================== #


def process_freeform(model, tokenizer, gen_client: OpenAI | None, judge_client: OpenAI, datasets: List[Path], qa_type: str) -> None:
    out = []
    total_score = 0
    total_n = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)

        print(f"[{qa_type}] {path.name}: {len(data)} questions", flush=True)

        for idx, item in enumerate(data):
            if idx % 5 == 0:
                print(f"[{qa_type}] {path.name} {idx}/{len(data)} ...", flush=True)

            q = item["question"]
            ref = item["answer"]

            ans = generate_answer(model, tokenizer, gen_client, q, qa_type, domain)
            score, just = grade_multiple(judge_client, q, ref, ans, domain)

            total_score += score
            out.append(
                {
                    "id": item["id"],
                    "type": item.get("type", qa_type),
                    "question": q,
                    "reference_answer": ref,
                    "model_answer": ans,
                    "correctness_score": score,
                }
            )

    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    out_path = OUTPUT_ROOT / f"{qa_type.lower()}.json"

    json.dump(
        {
            "type": qa_type,
            "results": out,
        },
        out_path.open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"[{qa_type}] Done → {out_path}", flush=True)


# ===================== MCQA ===================== #


def process_mcqa(model, tokenizer, gen_client: OpenAI | None, datasets: List[Path]) -> None:
    correct = 0
    total_n = 0
    out = []

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)

        print(f"[MCQA] {path.name}: {len(data)} questions", flush=True)

        for idx, item in enumerate(data):
            if idx % 5 == 0:
                print(f"[MCQA] {path.name} {idx}/{len(data)} ...", flush=True)

            prompt = (
                f"{domain_prompt(domain)}\n"
                "Choose the option that best matches the specification behavior.\n"
                "Reply ONLY with A/B/C/D.\n\n"
                f"Q: {item['question']}\n"
                + "\n".join(f"{k}: {v}" for k, v in item["options"].items())
            )

            if gen_client is not None:
                raw = generate_text_api(gen_client, prompt, temperature=0.1)
            else:
                raw = generate_text_local(model, tokenizer, prompt, temperature=0.1)
            pred = (raw.strip() or "A")[0].upper()
            gold = item["answer"].upper()
            is_correct = pred == gold
            correct += is_correct

            out.append(
                {
                    "id": item["id"],
                    "type": item.get("type", "MCQA"),
                    "question": item["question"],
                    "options": item["options"],
                    "reference_answer": gold,
                    "model_answer": pred,
                    "correct": is_correct,
                }
            )

    acc = correct / total_n if total_n else 0.0
    out_path = OUTPUT_ROOT / "mcqa.json"

    json.dump(
        {
            "type": "MCQA",
            "results": out,
        },
        out_path.open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"[MCQA] Accuracy = {acc:.3f}", flush=True)


# ===================== Main ===================== #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 answer generation for release.")
    parser.add_argument("--api-key", default=API_KEY, help="Generation API key (optional; use with --base-url).")
    parser.add_argument("--base-url", default=BASE_URL, help="Generation API base URL (optional; use with --api-key).")
    parser.add_argument("--model-name", default=MODEL_NAME, help="HF generation model (required if GEN_MODEL_NAME unset).")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Judge model name (required if JUDGE_MODEL unset).")
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL, help="Judge API base URL.")
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY, help="Judge API key (or set DEEPSEEK_API_KEY).")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output directory.")
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    if not args.judge_api_key:
        raise ValueError("Missing judge API key. Use --judge-api-key or set DEEPSEEK_API_KEY.")
    if not args.model_name:
        raise ValueError("Missing generation model. Use --model-name or set MODEL_NAME.")
    if not args.judge_model:
        raise ValueError("Missing judge model. Use --judge-model or set JUDGE_MODEL.")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_MODEL = args.judge_model
    JUDGE_BASE_URL = args.judge_base_url
    GRADING_PASSES = max(1, args.grading_passes)
    MAX_NEW_TOKENS = args.max_new_tokens
    OUTPUT_ROOT = Path(args.output_root)

    for p in DATASETS_AQA + DATASETS_EQA + DATASETS_MCQA:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    use_api_generation = bool(API_KEY and BASE_URL)
    if use_api_generation:
        print(f"Using API generation model: {MODEL_NAME}", flush=True)
        gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        model = None
        tokenizer = None
    else:
        print(f"Loading local HF model and tokenizer: {MODEL_NAME}", flush=True)
        gen_client = None
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()

    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    process_freeform(model, tokenizer, gen_client, judge_client, DATASETS_AQA, "AQA")
    process_freeform(model, tokenizer, gen_client, judge_client, DATASETS_EQA, "EQA")
    process_mcqa(model, tokenizer, gen_client, DATASETS_MCQA)

    print("[Stage-1] All experiments finished.", flush=True)
