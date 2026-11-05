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
from llama14b_local_client import build_hybrid_client

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

CHUNK_FILES = {
    "4G_NAS": Path("4Gnas_paragraph_chunks.jsonl"),
    "5G_NAS": Path("5Gnas_paragraph_chunks.jsonl"),
    "5G_RRC": Path("5Grrc_paragraph_chunks.jsonl"),
}

OUTPUT_ROOT = Path("stage2-llama14b-specfusion-nojudge")

TOP_K_CONTEXT = 8
BATCH_SIZE = 64
LOG_EVERY = 20
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
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


# ===================== Utilities ===================== #


def normalize_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return [t for t in text.split() if t]


def normalize_evidence(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


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
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = str(obj.get("text", "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            chunks.append({"text": text})
    return chunks


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
    q_freq = Counter(normalize_text(query))
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


def format_contexts(contexts: List[Dict], max_chars_per_chunk: int = 1600) -> str:
    out = []
    for i, ctx in enumerate(contexts, 1):
        text = ctx["text"]
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        out.append(f"[{i}]\n{text}")
    return "\n\n-----\n\n".join(out)


def extract_json(text: str) -> str:
    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block:
        return code_block.group(1).strip()
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return brace_match.group(0).strip()
    return re.sub(r"```json|```", "", text).strip()


def pad_contexts(contexts: List[Dict], k: int) -> List[Dict]:
    if len(contexts) >= k:
        return contexts[:k]
    return contexts + [{"text": ""}] * (k - len(contexts))


def load_all_items() -> Dict[str, List[Tuple[Path, Dict]]]:
    grouped: Dict[str, List[Tuple[Path, Dict]]] = {"AQA": [], "EQA": [], "MCQA": []}
    for qa_type, datasets in {"AQA": DATASETS_AQA, "EQA": DATASETS_EQA, "MCQA": DATASETS_MCQA}.items():
        for path in datasets:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                grouped[qa_type].append((path, item))
    return grouped


def call_judge_llm(judge_client: OpenAI, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = judge_client.chat.completions.create(
                model=JUDGE_MODEL,
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
    raw = call_judge_llm(
        judge_client,
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


def process_freeform(client, judge_client: OpenAI, items: List[Tuple[Path, Dict]], label: str, indices: Dict[str, Dict]) -> None:
    prompts = []
    all_items = []
    all_contexts = []

    for path, item in items:
        domain = detect_domain(path)
        idx = indices[domain]
        q = item["question"]
        contexts = retrieve(q, idx["chunks"], idx["bm25"], TOP_K_CONTEXT)
        task = (
            "This is an extractive question. Prefer exact text from context."
            if label == "EQA"
            else "This is an analytical question. Infer based on context and spec knowledge."
        )
        prompt = (
            f"{domain_prompt(domain)}\n{task}\n\n"
            f"Context:\n{format_contexts(contexts)}\n\n"
            f"Question:\n{q}\n\n"
            "Return JSON with keys: answer, evidence."
        )
        prompts.append(("You are a telecom expert.", prompt))
        all_items.append((path, item))
        all_contexts.append(contexts)

    print(f"[{label}] Running batch inference ({len(prompts)}, batch_size={BATCH_SIZE})", flush=True)
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        outputs.extend(client.batch_generate(batch, temperature=0.2, max_tokens=512))
        print(f"[{label}] LLM: {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", flush=True)

    results = []
    total_score = 0
    evidence_correct = 0
    evidence_total = 0
    for (path, item), raw, contexts in zip(all_items, outputs, all_contexts):
        domain = detect_domain(path)
        answer = raw
        evidence = ""
        try:
            parsed = json.loads(extract_json(raw))
            answer = str(parsed.get("answer", "")).strip() or raw
            evidence = str(parsed.get("evidence", "")).strip()
        except Exception:
            pass

        reference = str(item.get("answer", "")).strip()
        gold_evidence = item.get("evidence")
        score, just = grade_multiple(judge_client, item["question"], reference, answer, domain)
        total_score += score
        evidence_is_correct = bool(
            gold_evidence
            and normalize_evidence(evidence) == normalize_evidence(gold_evidence)
        )
        if gold_evidence:
            evidence_total += 1
            evidence_correct += int(evidence_is_correct)

        results.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "reference": reference,
                "gold_evidence": gold_evidence,
                "model_answer": answer,
                "evidence": evidence,
                "score": score,
                "evidence_correct": evidence_is_correct,
            }
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / f"{label.lower()}.json"
    out_path.write_text(
        json.dumps(
            {
                "type": label,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def process_mcqa(client, items: List[Tuple[Path, Dict]], indices: Dict[str, Dict]) -> None:
    prompts = []
    all_items = []
    all_contexts = []

    for path, item in items:
        domain = detect_domain(path)
        idx = indices[domain]
        q = item["question"]
        opts = item["options"]
        rq = q + " " + " ".join(str(v) for v in opts.values())
        contexts = retrieve(rq, idx["chunks"], idx["bm25"], TOP_K_CONTEXT)
        prompt = (
            f"{domain_prompt(domain)}\nUse context to answer.\n\n"
            f"Context:\n{format_contexts(contexts)}\n\n"
            f"Q: {q}\n"
            + "\n".join(f"{k}: {v}" for k, v in opts.items())
            + "\n\nReturn JSON with keys: answer, evidence. The answer must be A/B/C/D."
        )
        prompts.append(("You are a telecom expert.", prompt))
        all_items.append((path, item))
        all_contexts.append(contexts)

    print(f"[MCQA] Running batch inference ({len(prompts)}, batch_size={BATCH_SIZE})", flush=True)
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        outputs.extend(client.batch_generate(batch, temperature=0.1, max_tokens=256))
        print(f"[MCQA] LLM: {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", flush=True)

    results = []
    correct = 0
    evidence_correct = 0
    evidence_total = 0
    for (path, item), raw, contexts in zip(all_items, outputs, all_contexts):
        evidence = ""
        try:
            parsed = json.loads(extract_json(raw))
            answer_text = str(parsed.get("answer", "")).strip().upper()
            evidence = str(parsed.get("evidence", "")).strip()
        except Exception:
            answer_text = raw.upper()
        m = re.search(r"\b([A-D])\b", answer_text)
        pred = m.group(1) if m else "X"
        gold = str(item.get("answer", "")).strip().upper()
        gold_evidence = item.get("evidence")
        is_correct = pred == gold
        correct += int(is_correct)
        evidence_is_correct = bool(
            gold_evidence
            and normalize_evidence(evidence) == normalize_evidence(gold_evidence)
        )
        if gold_evidence:
            evidence_total += 1
            evidence_correct += int(evidence_is_correct)
        results.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "options": item["options"],
                "gold": gold,
                "gold_evidence": gold_evidence,
                "model_answer": pred,
                "evidence": evidence,
                "correct": is_correct,
                "evidence_correct": evidence_is_correct,
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
    global OUTPUT_ROOT, TOP_K_CONTEXT, BATCH_SIZE, LOG_EVERY, GRADING_PASSES

    parser = argparse.ArgumentParser(description="Stage-2 Llama14B chunk-RAG with LLM judge (release-safe).")
    parser.add_argument("--api-key", default=API_KEY, help="Optional generation API key; kept for config consistency.")
    parser.add_argument("--base-url", default=BASE_URL, help="Optional generation API base URL; kept for config consistency.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    if not args.judge_api_key:
        raise ValueError("Missing judge API key. Set JUDGE_API_KEY or pass --judge-api-key.")
    if not args.judge_base_url:
        raise ValueError("Missing judge base URL. Set JUDGE_BASE_URL or pass --judge-base-url.")
    if not args.judge_model:
        raise ValueError("Missing judge model. Set JUDGE_MODEL or pass --judge-model.")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_ROOT = Path(args.output_root)
    TOP_K_CONTEXT = max(1, args.top_k_context)
    BATCH_SIZE = max(1, args.batch_size)
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS_AQA + DATASETS_EQA + DATASETS_MCQA:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")
    for d, p in CHUNK_FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing chunk file for {d}: {p}")
    client = build_hybrid_client(
        local_model_name=MODEL_NAME,
        judge_api_key=JUDGE_API_KEY,
        judge_base_url=JUDGE_BASE_URL,
    )
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    indices: Dict[str, Dict] = {}
    for domain, p in CHUNK_FILES.items():
        c = load_chunks(p)
        indices[domain] = {"chunks": c, "bm25": build_bm25(c)}

    all_items = load_all_items()
    process_mcqa(client, all_items["MCQA"], indices)
    process_freeform(client, judge_client, all_items["AQA"], "AQA", indices)
    process_freeform(client, judge_client, all_items["EQA"], "EQA", indices)


if __name__ == "__main__":
    main()
