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

import numpy as np
from openai import OpenAI

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None
    TruncatedSVD = None

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

RAG_FILES = {
    "4G_NAS": Path("4Gnas_paragraph_rag_chunks.jsonl"),
    "5G_NAS": Path("5Gnas_paragraph_rag_chunks.jsonl"),
    "5G_RRC": Path("5Grrc_paragraph_rag_chunks.jsonl"),
}

KG_DIRS = {
    "4G_NAS": Path("4G_NAS_KG_JSON"),
    "5G_NAS": Path("5G_NAS_KG_JSON"),
    "5G_RRC": Path("5G_RRC_KG_JSON"),
}

OUTPUT_ROOT = Path("stage1-specfusion-specrag-kg")

TOP_K_CONTEXT = 8
HYBRID_ALPHA = 0.7
GRADING_PASSES = 1
LOG_EVERY = 20

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


# ===================== Retrieval ===================== #


def load_chunks(file_path: Path) -> List[Dict]:
    chunks: List[Dict] = []
    seen = set()
    if not file_path.exists():
        return chunks
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            clause_id = str(obj.get("clause_id", "")).strip()
            title = str(obj.get("clause_title", "")).strip()
            text = str(obj.get("text", "")).strip()
            citation_label = f"Clause {clause_id}" if clause_id else ""
            index_text = " ".join(x for x in (citation_label, title, text) if x)
            if not text or index_text in seen:
                continue
            seen.add(index_text)
            chunks.append(
                {
                    "component_type": "paragraph",
                    "citation_label": citation_label,
                    "clause_id": clause_id,
                    "clause_title": title,
                    "text": text,
                    "index_text": index_text,
                }
            )
    return chunks


def load_merged_para_chunks(domain: str) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    # Prefer citation-aware SpecAnchor paragraphs. The raw SpecFusion paragraph
    # file is retained only as a fallback when citation-aware chunks are absent.
    chunks = load_chunks(RAG_FILES[domain])
    if not chunks:
        chunks = load_chunks(CHUNK_FILES[domain])
    for chunk in chunks:
        key = chunk["index_text"]
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
    return merged


def load_kg(dir_path: Path) -> List[Dict]:
    chunks: List[Dict] = []
    seen = set()
    if not dir_path.exists():
        return chunks
    for fp in sorted(dir_path.glob("*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        clause_id = str(obj.get("clause_id", "")).strip()
        title = str(obj.get("title", "")).strip()
        raw_text = obj.get("text")
        text_parts: List[str] = []
        if title:
            text_parts.append(title)
        if isinstance(raw_text, list):
            text_parts.extend([str(x) for x in raw_text if str(x).strip()])
        elif isinstance(raw_text, str):
            text_parts.append(raw_text)
        text = "\n".join(text_parts).strip()
        if not text:
            continue
        uniq_key = f"{clause_id} {title} {text}".strip()
        if uniq_key in seen:
            continue
        seen.add(uniq_key)
        citation_label = f"Clause {clause_id}" if clause_id else ""
        index_text = " ".join(x for x in (citation_label, title, text) if x)
        chunks.append(
            {
                "component_type": "kg",
                "citation_label": citation_label,
                "clause_id": clause_id,
                "clause_title": title,
                "text": text,
                "index_text": index_text,
            }
        )
    return chunks


def build_bm25(chunks: List[Dict]) -> Dict:
    tokenized = [normalize_text(c["index_text"]) for c in chunks]
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


def build_dense_index(chunks: List[Dict]) -> Dict:
    if TfidfVectorizer is None or TruncatedSVD is None:
        raise RuntimeError("scikit-learn is required for hybrid retrieval.")
    corpus = [c["index_text"] for c in chunks]
    if not corpus:
        return {"vectorizer": None, "svd": None, "embeddings": np.zeros((0, 1))}

    vectorizer = TfidfVectorizer(max_features=30000)
    x = vectorizer.fit_transform(corpus)
    max_components = min(x.shape[0], x.shape[1]) - 1
    if max_components < 1:
        return {"vectorizer": vectorizer, "svd": None, "embeddings": x.toarray()}
    n_components = min(256, max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    x_svd = svd.fit_transform(x)
    emb = x_svd / (np.linalg.norm(x_svd, axis=1, keepdims=True) + 1e-8)
    return {"vectorizer": vectorizer, "svd": svd, "embeddings": emb}


def dense_score(query: str, dense_index: Dict) -> List[float]:
    vec = dense_index["vectorizer"]
    svd = dense_index["svd"]
    emb = dense_index["embeddings"]
    if vec is None or emb.shape[0] == 0:
        return []
    q = vec.transform([query])
    q_svd = svd.transform(q) if svd is not None else q.toarray()
    q_unit = (q_svd / (np.linalg.norm(q_svd) + 1e-8)).reshape(-1)
    return (emb @ q_unit).tolist()


def hybrid_retrieve(query: str, chunks: List[Dict], bm25_index: Dict, dense_index: Dict, k: int, alpha: float) -> List[Dict]:
    if not chunks:
        return []

    sparse = bm25_score(query, bm25_index)
    sparse_max = max(sparse) if sparse else 0.0
    sparse_norm = [s / sparse_max for s in sparse] if sparse_max > 0 else [0.0] * len(chunks)

    dense = dense_score(query, dense_index)
    dense_norm = [(d + 1.0) / 2.0 for d in dense] if dense else [0.0] * len(chunks)

    scores = [alpha * ssp + (1.0 - alpha) * sde for ssp, sde in zip(sparse_norm, dense_norm)]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in ranked[:k]]


def format_contexts(contexts: List[Dict]) -> str:
    formatted = []
    for i, chunk in enumerate(contexts, 1):
        source = chunk.get("component_type", "component").upper()
        label = chunk.get("citation_label", "") or "Unlabeled"
        title = chunk.get("clause_title", "")
        heading = f"[{i}] [{source}] {label}" + (f" - {title}" if title else "")
        formatted.append(f"{heading}\n{chunk.get('text', '')}")
    return "\n\n-----\n\n".join(formatted)


# ===================== Generation / Judge ===================== #


def generate_answer(gen_client: OpenAI, question: str, contexts: List[Dict], qa_type: str, domain: str) -> str:
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
        "The provided context consists of raw text excerpts from the specification and KG clauses.\n"
        "Use the context ONLY if it adds concrete constraints, conditions, or exceptions.\n"
        "If the context is vague, generic, or inconsistent, you may ignore it.\n\n"
        f"{task_inst}\n\n"
        f"Context:\n{format_contexts(contexts)}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    return call_llm(gen_client, MODEL_NAME, [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=512)


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


def process_freeform(gen_client: OpenAI, judge_client: OpenAI, datasets: List[Path], qa_type: str, indices: Dict[str, Dict]) -> None:
    out = []
    total_score = 0
    total_n = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        idx = indices[domain]

        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[{qa_type}] {path.name} {i}/{len(data)}", flush=True)

            q = item["question"]
            ref = item["answer"]
            ctxs = hybrid_retrieve(
                q,
                idx["chunks"],
                idx["bm25"],
                idx["dense"],
                TOP_K_CONTEXT,
                HYBRID_ALPHA,
            )

            ans = generate_answer(gen_client, q, ctxs, qa_type, domain)
            score, just = grade_multiple(judge_client, q, ref, ans, domain)
            total_score += score

            out.append({
                "id": item["id"],
                "type": item.get("type", qa_type),
                "question": q,
                "reference_answer": ref,
                "model_answer": ans,
                "correctness_score": score,
            })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / f"{qa_type.lower()}.json"
    out_path.write_text(
        json.dumps({
            "type": qa_type,
            "results": out,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def process_mcqa(gen_client: OpenAI, datasets: List[Path], indices: Dict[str, Dict]) -> None:
    out = []
    correct = 0
    total_n = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        idx = indices[domain]

        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[MCQA] {path.name} {i}/{len(data)}", flush=True)

            retrieval_query = item["question"] + " " + " ".join(str(v) for v in item["options"].values())
            contexts = hybrid_retrieve(
                retrieval_query,
                idx["chunks"],
                idx["bm25"],
                idx["dense"],
                TOP_K_CONTEXT,
                HYBRID_ALPHA,
            )
            prompt = (
                f"{domain_prompt(domain)}\n"
                "Use the retrieved evidence to choose the option that best matches the specification behavior.\n"
                "Reply ONLY with A/B/C/D.\n\n"
                f"Retrieved Evidence:\n{format_contexts(contexts)}\n\n"
                f"Q: {item['question']}\n"
                + "\n".join(f"{k}: {v}" for k, v in item["options"].items())
            )
            raw = call_llm(gen_client, MODEL_NAME, [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=64)
            m = re.search(r"\b([A-D])\b", raw.upper())
            pred = m.group(1) if m else "X"
            gold = str(item["answer"]).strip().upper()
            is_correct = pred == gold
            correct += int(is_correct)
            out.append({
                "id": item["id"],
                "type": item.get("type", "MCQA"),
                "question": item["question"],
                "options": item["options"],
                "reference_answer": gold,
                "model_answer": pred,
                "correct": is_correct,
            })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / "mcqa.json"
    out_path.write_text(
        json.dumps({
            "type": "MCQA",
            "results": out,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    global API_KEY, BASE_URL, JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL, MODEL_NAME
    global OUTPUT_ROOT, TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY

    parser = argparse.ArgumentParser(description="Stage-1 KG + paragraph hybrid retrieval (release-safe).")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME (or pass args).")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model:
        raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL (or pass args).")
    if TfidfVectorizer is None or TruncatedSVD is None:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_ROOT = Path(args.output_root)
    TOP_K_CONTEXT = max(1, args.top_k_context)
    HYBRID_ALPHA = max(0.0, min(1.0, args.hybrid_alpha))
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS_AQA + DATASETS_EQA + DATASETS_MCQA:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    indices: Dict[str, Dict] = {}
    for domain in CHUNK_FILES:
        if not CHUNK_FILES[domain].exists():
            raise FileNotFoundError(f"Missing chunk file for {domain}: {CHUNK_FILES[domain]}")
        if not RAG_FILES[domain].exists():
            raise FileNotFoundError(f"Missing rag file for {domain}: {RAG_FILES[domain]}")
        if not KG_DIRS[domain].exists():
            raise FileNotFoundError(f"Missing KG dir for {domain}: {KG_DIRS[domain]}")

        para_chunks = load_merged_para_chunks(domain)
        kg_chunks = load_kg(KG_DIRS[domain])
        chunks = para_chunks + kg_chunks
        indices[domain] = {
            "chunks": chunks,
            "bm25": build_bm25(chunks),
            "dense": build_dense_index(chunks),
        }

    gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    process_freeform(gen_client, judge_client, DATASETS_AQA, "AQA", indices)
    process_freeform(gen_client, judge_client, DATASETS_EQA, "EQA", indices)
    process_mcqa(gen_client, DATASETS_MCQA, indices)


if __name__ == "__main__":
    main()
