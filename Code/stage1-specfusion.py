from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

# ===================== Configuration ===================== #

DATA_ROOT = Path("")
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

CHUNK_FILES = {
    "4G_NAS": Path("4Gnas_paragraph_chunks.jsonl"),
    "5G_NAS": Path("5Gnas_paragraph_chunks.jsonl"),
    "5G_RRC": Path("5Grrc_paragraph_chunks.jsonl"),
}

OUTPUT_ROOT = Path("stage1-specfusion")

TOP_K_CONTEXT = 8
GRADING_PASSES = 1
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


def normalize_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return [t for t in text.split() if t]


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


def call_llm(client: OpenAI, model_name: str, messages: List[Dict], temperature: float) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            sleep_s = (RETRY_BASE_SLEEP * (2 ** attempt)) + random.random() * RETRY_JITTER
            time.sleep(sleep_s)
    return f"[ERROR] {repr(last_err)}"


# =========================================================
# 1. Load Paragraph Chunks (TEXT ONLY)
# =========================================================


def load_chunks(file_path: Path) -> List[Dict]:
    chunks = []
    seen = set()
    if not file_path.exists():
        return chunks

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            chunks.append({"text": text})

    print(f"[RAG] Loaded {len(chunks)} chunks from {file_path}", flush=True)
    return chunks


# =========================================================
# 2. BM25 (Text-only)
# =========================================================


def build_bm25(chunks: List[Dict]) -> Dict:
    tokenized = [normalize_text(c["text"]) for c in chunks]
    df = Counter()
    for toks in tokenized:
        for w in set(toks):
            df[w] += 1

    n_docs = len(chunks) or 1
    avgdl = sum(len(t) for t in tokenized) / n_docs

    idf = {w: math.log(1 + (n_docs - f + 0.5) / (f + 0.5)) for w, f in df.items()}
    return {"tokenized": tokenized, "idf": idf, "avgdl": avgdl}


def bm25_score(query: str, bm25_index: Dict) -> List[float]:
    q_tokens = normalize_text(query)
    q_freq = Counter(q_tokens)

    out = []
    for toks in bm25_index["tokenized"]:
        dl = len(toks)
        tf = Counter(toks)
        score = 0.0
        for term, qf in q_freq.items():
            if term not in bm25_index["idf"]:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf_t = bm25_index["idf"][term]
            denom = f + 1.5 * (1 - 0.75 + 0.75 * dl / bm25_index["avgdl"])
            score += idf_t * (f * 2.5 / denom) * qf
        out.append(score)
    return out


def retrieve(question: str, chunks: List[Dict], bm25_index: Dict, k: int) -> List[Dict]:
    if not chunks:
        return []
    scores = bm25_score(question, bm25_index)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in ranked[:k]]


# =========================================================
# 3. Prompting
# =========================================================


def format_contexts(contexts: List[Dict]) -> str:
    return "\n\n-----\n\n".join(c["text"] for c in contexts)


def generate_answer(client: OpenAI, question: str, contexts: List[Dict], qa_type: str, domain: str) -> str:
    ctx = format_contexts(contexts) if contexts else ""

    if qa_type == "EQA":
        task_inst = (
            "This is an extractive question.\n"
            "- If the context contains an exact statement answering the question, quote it.\n"
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
        "The provided context consists of raw text excerpts from the specification.\n"
        "Use the context ONLY if it adds concrete constraints, conditions, or exceptions.\n"
        "If the context is vague, generic, or inconsistent, you may ignore it.\n\n"
        f"{task_inst}\n\n"
    )

    if ctx:
        prompt += f"Context:\n{ctx}\n\n"
    prompt += f"Question:\n{question}\n\nAnswer:"

    return call_llm(client, MODEL_NAME, [{"role": "user", "content": prompt}], temperature=0.2)


# =========================================================
# 4. LLM-as-Judge
# =========================================================


def judge_answer(client: OpenAI, question: str, reference: str, candidate: str, domain: str) -> Tuple[int, str]:
    prompt = (
        "Grade this answer.\n\n"
        "Score 2 = Correct\n"
        "Score 1 = Partially correct\n"
        "Score 0 = Incorrect or contradicts spec\n\n"
        f"Question:\n{question}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Model answer:\n{candidate}\n\n"
        "Respond ONLY as JSON with keys score and justification."
    )
    raw = call_llm(
        client,
        JUDGE_MODEL,
        [
            {"role": "system", "content": domain_prompt(domain)},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        obj = json.loads(raw)
        score = int(obj.get("score", 0))
        justification = str(obj.get("justification", "")).strip()
    except Exception:
        m = re.search(r"\b([0-2])\b", raw)
        score = int(m.group(1)) if m else 0
        justification = raw
    return max(0, min(score, 2)), justification


def grade_multiple(client: OpenAI, q: str, ref: str, ans: str, domain: str) -> Tuple[int, str]:
    scores = []
    justification = ""
    for _ in range(max(1, GRADING_PASSES)):
        s, j = judge_answer(client, q, ref, ans, domain)
        scores.append(s)
        justification = justification or j
    return max(set(scores), key=scores.count), justification


# =========================================================
# 5. Freeform QA (AQA / EQA)
# =========================================================


def process_freeform(
    gen_client: OpenAI,
    judge_client: OpenAI,
    datasets: List[Path],
    qa_type: str,
    indices: Dict[str, Dict],
    log_every: int,
) -> None:
    out = []
    total_score = 0
    total_n = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        chunks = indices[domain]["chunks"]
        bm25_index = indices[domain]["bm25"]

        print(f"[{qa_type}] {path.name}: {len(data)} questions", flush=True)

        for idx, item in enumerate(data, 1):
            if log_every > 0 and (idx % log_every == 0 or idx == len(data)):
                print(f"[{qa_type}] {path.name} {idx}/{len(data)} ...", flush=True)

            q = item["question"]
            ref = item["answer"]
            ctxs = retrieve(q, chunks, bm25_index, TOP_K_CONTEXT)
            ans = generate_answer(gen_client, q, ctxs, qa_type, domain)
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
    out_path.write_text(
        json.dumps(
            {
                "type": qa_type,
                "results": out,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[{qa_type}] Done -> {out_path}", flush=True)


# =========================================================
# 6. MCQA
# =========================================================


def process_mcqa(
    gen_client: OpenAI,
    datasets: List[Path],
    indices: Dict[str, Dict],
    log_every: int,
) -> None:
    correct = 0
    total_n = 0
    out = []

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        chunks = indices[domain]["chunks"]
        bm25_index = indices[domain]["bm25"]

        print(f"[MCQA] {path.name}: {len(data)} questions", flush=True)

        for idx, item in enumerate(data, 1):
            if log_every > 0 and (idx % log_every == 0 or idx == len(data)):
                print(f"[MCQA] {path.name} {idx}/{len(data)} ...", flush=True)

            retrieval_query = item["question"] + " " + " ".join(
                str(value) for value in item["options"].values()
            )
            contexts = retrieve(retrieval_query, chunks, bm25_index, TOP_K_CONTEXT)
            prompt = (
                f"{domain_prompt(domain)}\n"
                "Use the retrieved specification chunks to choose the option that best matches the specification behavior.\n"
                "Reply ONLY with A/B/C/D.\n\n"
                f"Context:\n{format_contexts(contexts)}\n\n"
                f"Q: {item['question']}\n"
                + "\n".join(f"{k}: {v}" for k, v in item["options"].items())
            )
            raw = call_llm(gen_client, MODEL_NAME, [{"role": "user", "content": prompt}], temperature=0.1)
            m = re.search(r"\b([A-D])\b", raw.upper())
            pred = m.group(1) if m else (raw.strip()[:1].upper() if raw.strip() else "A")
            gold = str(item["answer"]).strip().upper()
            is_correct = pred == gold
            correct += int(is_correct)

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
    out_path.write_text(
        json.dumps(
            {
                "type": "MCQA",
                "results": out,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[MCQA] Accuracy = {acc:.3f}", flush=True)
    print(f"[MCQA] Done -> {out_path}", flush=True)


# =========================================================
# 7. Main
# =========================================================


def main() -> None:
    global API_KEY, BASE_URL, MODEL_NAME
    global JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL
    global OUTPUT_ROOT, GRADING_PASSES, TOP_K_CONTEXT

    parser = argparse.ArgumentParser(description="Stage-1 RAG QA with Qwen (release-safe).")
    parser.add_argument("--api-key", default=API_KEY, help="Qwen/OpenAI-compatible API key.")
    parser.add_argument("--base-url", default=BASE_URL, help="Qwen/OpenAI-compatible base URL.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Generation model name.")
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY, help="Judge API key.")
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL, help="Judge API base URL.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Judge model name.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output directory.")
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set it in code or pass --api-key.")
    if not args.base_url:
        raise ValueError("Missing base URL. Set it in code or pass --base-url.")
    if not args.model_name:
        raise ValueError("Missing model name. Set it in code or pass --model-name.")
    if not args.judge_api_key:
        raise ValueError("Missing judge API key. Set it in code or pass --judge-api-key.")
    if not args.judge_base_url:
        raise ValueError("Missing judge base URL. Set it in code or pass --judge-base-url.")
    if not args.judge_model:
        raise ValueError("Missing judge model. Set it in code or pass --judge-model.")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_ROOT = Path(args.output_root)
    GRADING_PASSES = max(1, args.grading_passes)
    TOP_K_CONTEXT = max(1, args.top_k_context)

    for p in DATASETS_AQA + DATASETS_EQA + DATASETS_MCQA:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset: {p}")
    for k, p in CHUNK_FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing chunk file for {k}: {p}")

    print("[SpecFusion] Starting Stage-1 RAG (Qwen release)...", flush=True)

    gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    indices: Dict[str, Dict] = {}
    for domain, chunk_path in CHUNK_FILES.items():
        chunks = load_chunks(chunk_path)
        bm25_index = build_bm25(chunks)
        indices[domain] = {"chunks": chunks, "bm25": bm25_index}

    process_freeform(gen_client, judge_client, DATASETS_AQA, "AQA", indices, args.log_every)
    process_freeform(gen_client, judge_client, DATASETS_EQA, "EQA", indices, args.log_every)
    process_mcqa(gen_client, DATASETS_MCQA, indices, args.log_every)

    print("[SpecFusion] All Stage-1 experiments finished.", flush=True)


if __name__ == "__main__":
    main()
