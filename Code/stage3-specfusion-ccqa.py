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

DATA_ROOT = Path("")
DATASETS = [DATA_ROOT / "4G_NAS_stage3-CCQA.json", DATA_ROOT / "5G_NAS_stage3-CCQA.json", DATA_ROOT / "5G_RRC_stage3-CCQA.json"]
CHUNK_FILES = {"4G_NAS": Path("4Gnas_paragraph_chunks.jsonl"), "5G_NAS": Path("5Gnas_paragraph_chunks.jsonl"), "5G_RRC": Path("5Grrc_paragraph_chunks.jsonl")}
OUTPUT_FILE = Path("stage3-specfusion-ccqa.json")
TOP_K_CONTEXT, GRADING_PASSES, LOG_EVERY = 8, 1, 20
MAX_RETRIES, RETRY_BASE_SLEEP, RETRY_JITTER = 6, 1.2, 0.5
API_KEY = BASE_URL = MODEL_NAME = ""
JUDGE_API_KEY = JUDGE_BASE_URL = JUDGE_MODEL = ""


def normalize_text(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split()


def normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def detect_domain(path: Path) -> str:
    for domain in CHUNK_FILES:
        if path.name.startswith(domain):
            return domain
    raise ValueError(f"Unknown dataset domain: {path}")


def domain_prompt(domain: str) -> str:
    return {"4G_NAS": "You are a 4G NAS expert (3GPP TS 24.301).", "5G_NAS": "You are a 5G NAS expert (3GPP TS 24.501).", "5G_RRC": "You are a 5G RRC expert (3GPP TS 38.331)."}[domain]


def call_llm(client: OpenAI, model: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return (response.choices[0].message.content or "").strip()
        except Exception as error:
            last_error = error
            time.sleep(RETRY_BASE_SLEEP * (2**attempt) + random.random() * RETRY_JITTER)
    return f"[ERROR] {last_error!r}"


def load_chunks(path: Path) -> List[Dict]:
    chunks, seen = [], set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                text = str(json.loads(line).get("text", "")).strip()
            except Exception:
                continue
            if text and text not in seen:
                seen.add(text)
                chunks.append({"text": text})
    return chunks


def build_bm25(chunks: List[Dict]) -> Dict:
    tokenized = [normalize_text(chunk["text"]) for chunk in chunks]
    df = Counter(token for tokens in tokenized for token in set(tokens))
    count = len(chunks) or 1
    avgdl = sum(map(len, tokenized)) / count
    idf = {token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5)) for token, frequency in df.items()}
    return {"tokenized": tokenized, "idf": idf, "avgdl": avgdl}


def retrieve(query: str, chunks: List[Dict], index: Dict, top_k: int) -> List[Dict]:
    query_frequency, scores = Counter(normalize_text(query)), []
    for tokens in index["tokenized"]:
        term_frequency, length, score = Counter(tokens), len(tokens), 0.0
        for term, query_count in query_frequency.items():
            frequency = term_frequency.get(term, 0)
            if not frequency or term not in index["idf"]:
                continue
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / index["avgdl"])
            score += index["idf"][term] * frequency * 2.5 / denominator * query_count
        scores.append(score)
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return [chunks[position] for position, _ in ranked[:top_k]]


def format_contexts(contexts: List[Dict]) -> str:
    return "\n\n-----\n\n".join(f"[{number}]\n{chunk['text'][:1800]}" for number, chunk in enumerate(contexts, 1))


def generate_answer(client: OpenAI, model: str, question: str, contexts: List[Dict], domain: str) -> Tuple[str, str]:
    prompt = f"{domain_prompt(domain)}\nAnswer the cross-clause question by combining all relevant retrieved specification chunks.\nReturn strict JSON with keys answer and evidence. Evidence should contain the most likely 3GPP clause citations supported by the context.\n\nRetrieved Context:\n{format_contexts(contexts)}\n\nQuestion:\n{question}"
    raw = call_llm(client, model, [{"role": "user", "content": prompt}], 0.2, 768)
    try:
        obj = json.loads(re.sub(r"```json|```", "", raw).strip())
        return str(obj.get("answer", "")).strip() or raw, str(obj.get("evidence", "")).strip()
    except Exception:
        return raw, ""


def judge_answer(client: OpenAI, model: str, question: str, reference: str, answer: str, domain: str) -> int:
    prompt = f"Grade this answer against the reference. Score 2 = correct, 1 = partially correct, 0 = incorrect. Return strict JSON with keys score and justification.\n\nQuestion:\n{question}\n\nReference:\n{reference}\n\nModel answer:\n{answer}"
    raw = call_llm(client, model, [{"role": "system", "content": domain_prompt(domain)}, {"role": "user", "content": prompt}], 0.0, 256)
    try:
        score = int(json.loads(re.sub(r"```json|```", "", raw).strip()).get("score", 0))
    except Exception:
        match = re.search(r"\b([0-2])\b", raw)
        score = int(match.group(1)) if match else 0
    return max(0, min(score, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 CCQA with BM25 SpecFusion chunks")
    parser.add_argument("--api-key", default=API_KEY); parser.add_argument("--base-url", default=BASE_URL); parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY); parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL); parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output-file", default=str(OUTPUT_FILE)); parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES); parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()
    if not args.api_key or not args.base_url or not args.model_name: raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME.")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model: raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL.")
    for path in DATASETS + list(CHUNK_FILES.values()):
        if not path.exists(): raise FileNotFoundError(path)
    generator, judge = OpenAI(api_key=args.api_key, base_url=args.base_url), OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    indices = {}
    for domain, path in CHUNK_FILES.items():
        chunks = load_chunks(path); indices[domain] = {"chunks": chunks, "bm25": build_bm25(chunks)}
    results = []
    for path in DATASETS:
        data, domain = json.loads(path.read_text(encoding="utf-8")), detect_domain(path)
        for position, item in enumerate(data, 1):
            if position % max(1, args.log_every) == 0 or position == len(data): print(f"[CCQA] {path.name} {position}/{len(data)}", flush=True)
            question, reference, gold = item["question"], item["answer"], item.get("evidence", "")
            index = indices[domain]; contexts = retrieve(question, index["chunks"], index["bm25"], max(1, args.top_k_context))
            answer, evidence = generate_answer(generator, args.model_name, question, contexts, domain)
            scores = [judge_answer(judge, args.judge_model, question, reference, answer, domain) for _ in range(max(1, args.grading_passes))]
            results.append({"id": item.get("id"), "question": question, "reference": reference, "gold_evidence": gold, "model_answer": answer, "evidence": evidence, "score": max(set(scores), key=scores.count), "evidence_correct": bool(gold and normalize_evidence(evidence) == normalize_evidence(gold))})
    output = Path(args.output_file); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"type": "CCQA", "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
