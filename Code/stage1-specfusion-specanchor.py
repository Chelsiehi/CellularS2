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

COMPONENT_DIRS = {
    "4G_NAS": {"table": Path("SpecFusion/4G_NAS_Table"), "figure": Path("SpecFusion/4G_NAS_Figure")},
    "5G_NAS": {"table": Path("SpecFusion/5G_NAS_Table"), "figure": Path("SpecFusion/5G_NAS_Figure")},
    "5G_RRC": {"table": Path("SpecFusion/5G_RRC_Table"), "figure": Path("SpecFusion/5G_RRC_Figure")},
}

OUTPUT_ROOT = Path("stage1-specfusion-specrag")

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


# ===================== Retrieval (SpecRAG) ===================== #


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
            clause_id = str(obj.get("clause_id", "")).strip()
            clause_title = str(obj.get("clause_title", "")).strip()
            text = str(obj.get("text", "")).strip()
            citation_label = f"Clause {clause_id}" if clause_id else ""
            index_text = " ".join(x for x in (citation_label, clause_title, text) if x)
            if not text or index_text in seen:
                continue
            seen.add(index_text)
            chunks.append(
                {
                    "component_type": "subclause",
                    "citation_label": citation_label,
                    "clause_id": clause_id,
                    "clause_title": clause_title,
                    "text": text,
                    "index_text": index_text,
                }
            )

    print(f"[RAG] Loaded {len(chunks)} chunks from {file_path}", flush=True)
    return chunks


def load_structured_components(directory: Path, component_type: str) -> List[Dict]:
    components: List[Dict] = []
    if not directory.exists():
        return components
    for file_path in sorted(directory.glob("*.json")):
        try:
            obj = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if component_type == "table":
            title = str(obj.get("table_title", "")).strip()
            citation_label = title.split(":", 1)[0].strip() if title else file_path.stem
        else:
            figure_id = str(obj.get("figure_id", "")).strip()
            title = str(obj.get("title", "")).strip()
            citation_label = f"Figure {figure_id}" if figure_id else file_path.stem
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        index_text = " ".join(x for x in (citation_label, title, body) if x)
        components.append(
            {
                "component_type": component_type,
                "citation_label": citation_label,
                "clause_id": "",
                "clause_title": title,
                "text": body,
                "index_text": index_text,
            }
        )
    return components


def load_merged_chunks_for_domain(domain: str) -> List[Dict]:
    seen = set()
    merged: List[Dict] = []

    # The citation-aware RAG chunks represent subclauses. Raw paragraph chunks
    # are used only as a fallback because they do not carry citation labels.
    subclauses = load_chunks(RAG_FILES[domain])
    if not subclauses:
        subclauses = load_chunks(CHUNK_FILES[domain])
    components = list(subclauses)
    for component_type, directory in COMPONENT_DIRS[domain].items():
        components.extend(load_structured_components(directory, component_type))

    for c in components:
        key = c["index_text"].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(c)

    print(f"[RAG] Domain={domain} merged chunks: {len(merged)}", flush=True)
    return merged


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
    q_tokens = normalize_text(query)
    q_freq = Counter(q_tokens)
    scores = []

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
        scores.append(score)
    return scores


def build_dense_index(chunks: List[Dict]) -> Dict:
    if TfidfVectorizer is None or TruncatedSVD is None:
        raise RuntimeError("scikit-learn is required for SpecRAG dense retrieval (TF-IDF + SVD).")

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
    norms = np.linalg.norm(x_svd, axis=1, keepdims=True) + 1e-8
    emb = x_svd / norms
    return {"vectorizer": vectorizer, "svd": svd, "embeddings": emb}


def dense_score(query: str, dense_index: Dict) -> List[float]:
    vec = dense_index["vectorizer"]
    svd = dense_index["svd"]
    emb = dense_index["embeddings"]
    if vec is None or svd is None or emb.shape[0] == 0:
        return []

    q_tf = vec.transform([query])
    q_svd = svd.transform(q_tf) if svd is not None else q_tf.toarray()
    q_norm = np.linalg.norm(q_svd) + 1e-8
    q_unit = (q_svd / q_norm).reshape(-1)
    sims = emb @ q_unit
    return sims.tolist()


def hybrid_retrieve(query: str, chunks: List[Dict], bm25_index: Dict, dense_index: Dict, k: int, alpha: float) -> List[Dict]:
    if not chunks:
        return []

    sparse = bm25_score(query, bm25_index)
    max_sparse = max(sparse) if sparse else 0.0
    sparse_norm = [s / max_sparse for s in sparse] if max_sparse > 0 else [0.0] * len(chunks)

    dense = dense_score(query, dense_index)
    dense_norm = [(d + 1.0) / 2.0 for d in dense] if dense else [0.0] * len(chunks)

    hybrid_scores = [alpha * ssp + (1.0 - alpha) * sde for ssp, sde in zip(sparse_norm, dense_norm)]
    ranked = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in ranked[:k]]


# ===================== Answer Generation ===================== #


def format_contexts(contexts: List[Dict], max_chars_per_chunk: int = 1600) -> str:
    out = []
    for i, c in enumerate(contexts, 1):
        txt = c.get("text", "")
        if len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk] + "..."
        label = c.get("citation_label", "") or "Unlabeled component"
        title = c.get("clause_title", "")
        heading = f"[{i}] {label}" + (f" - {title}" if title and title not in label else "")
        out.append(f"{heading}\n{txt}")
    return "\n\n-----\n\n".join(out)


def generate_answer(gen_client: OpenAI, question: str, contexts: List[Dict], qa_type: str, domain: str) -> str:
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

    return call_llm(
        gen_client,
        model_name=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )


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
        model_name=JUDGE_MODEL,
        messages=[
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
        justification = str(obj.get("justification", "")).strip()
    except Exception:
        m = re.search(r"\b([0-2])\b", raw)
        score = int(m.group(1)) if m else 0
        justification = raw

    return max(0, min(score, 2)), justification


def grade_multiple(judge_client: OpenAI, q: str, ref: str, ans: str, domain: str) -> Tuple[int, str]:
    scores = []
    justification = ""
    for _ in range(max(1, GRADING_PASSES)):
        s, j = judge_answer(judge_client, q, ref, ans, domain)
        scores.append(s)
        justification = justification or j
    return max(set(scores), key=scores.count), justification


# ===================== Freeform QA (AQA / EQA) ===================== #


def process_freeform(
    gen_client: OpenAI,
    judge_client: OpenAI,
    datasets: List[Path],
    qa_type: str,
    indices: Dict[str, Dict],
) -> None:
    out = []
    total_score = 0
    total_n = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        idx_obj = indices[domain]

        print(f"[{qa_type}] {path.name}: {len(data)} questions", flush=True)

        for idx, item in enumerate(data, 1):
            if idx % LOG_EVERY == 0 or idx == len(data):
                print(f"[{qa_type}] {path.name} {idx}/{len(data)} ...", flush=True)

            q = item["question"]
            ref = item["answer"]
            ctxs = hybrid_retrieve(
                query=q,
                chunks=idx_obj["chunks"],
                bm25_index=idx_obj["bm25"],
                dense_index=idx_obj["dense"],
                k=TOP_K_CONTEXT,
                alpha=HYBRID_ALPHA,
            )
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
    json.dump(
        {
            "type": qa_type,
            "results": out,
        },
        out_path.open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"[{qa_type}] Done -> {out_path}", flush=True)


# ===================== MCQA ===================== #


def process_mcqa(gen_client: OpenAI, datasets: List[Path], indices: Dict[str, Dict]) -> None:
    correct = 0
    total_n = 0
    out = []

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        idx_obj = indices[domain]

        print(f"[MCQA] {path.name}: {len(data)} questions", flush=True)

        for idx, item in enumerate(data, 1):
            if idx % LOG_EVERY == 0 or idx == len(data):
                print(f"[MCQA] {path.name} {idx}/{len(data)} ...", flush=True)

            retrieval_query = item["question"] + " " + " ".join(str(v) for v in item["options"].values())
            contexts = hybrid_retrieve(
                query=retrieval_query,
                chunks=idx_obj["chunks"],
                bm25_index=idx_obj["bm25"],
                dense_index=idx_obj["dense"],
                k=TOP_K_CONTEXT,
                alpha=HYBRID_ALPHA,
            )
            prompt = (
                f"{domain_prompt(domain)}\n"
                "Use the cited specification components below to choose the option that best matches the specification behavior.\n"
                "Reply ONLY with A/B/C/D.\n\n"
                f"Retrieved Evidence:\n{format_contexts(contexts)}\n\n"
                f"Q: {item['question']}\n"
                + "\n".join(f"{k}: {v}" for k, v in item["options"].items())
            )

            raw = call_llm(
                gen_client,
                model_name=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=64,
            )
            m = re.search(r"\b([A-D])\b", raw.upper())
            pred = m.group(1) if m else "X"
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
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
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
    print(f"[MCQA] Done -> {out_path}", flush=True)


# ===================== Main ===================== #


def main() -> None:
    global API_KEY, BASE_URL, MODEL_NAME
    global JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL
    global OUTPUT_ROOT, TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY

    parser = argparse.ArgumentParser(description="Stage-1 SpecFusion+SpecRAG generation (release-safe).")
    parser.add_argument("--api-key", default=API_KEY, help="Generation API key.")
    parser.add_argument("--base-url", default=BASE_URL, help="Generation API base URL.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Generation model name.")
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY, help="Judge API key.")
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL, help="Judge API base URL.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Judge model name.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output directory.")
    parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing generation API key. Set API_KEY or pass --api-key.")
    if not args.base_url:
        raise ValueError("Missing generation base URL. Set BASE_URL or pass --base-url.")
    if not args.model_name:
        raise ValueError("Missing generation model. Set MODEL_NAME or pass --model-name.")
    if not args.judge_api_key:
        raise ValueError("Missing judge API key. Set JUDGE_API_KEY or pass --judge-api-key.")
    if not args.judge_base_url:
        raise ValueError("Missing judge base URL. Set JUDGE_BASE_URL or pass --judge-base-url.")
    if not args.judge_model:
        raise ValueError("Missing judge model. Set JUDGE_MODEL or pass --judge-model.")
    if TfidfVectorizer is None or TruncatedSVD is None:
        raise RuntimeError("scikit-learn is required. Please install: pip install scikit-learn")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_ROOT = Path(args.output_root)
    TOP_K_CONTEXT = args.top_k_context
    HYBRID_ALPHA = min(1.0, max(0.0, args.hybrid_alpha))
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS_AQA + DATASETS_EQA + DATASETS_MCQA:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    for domain in CHUNK_FILES:
        if not CHUNK_FILES[domain].exists():
            raise FileNotFoundError(f"Missing chunk file for {domain}: {CHUNK_FILES[domain]}")
        if not RAG_FILES[domain].exists():
            raise FileNotFoundError(f"Missing rag file for {domain}: {RAG_FILES[domain]}")

    print("[Stage-1] Building SpecRAG indices...", flush=True)
    indices: Dict[str, Dict] = {}
    for domain in CHUNK_FILES:
        chunks = load_merged_chunks_for_domain(domain)
        bm25_index = build_bm25(chunks)
        dense_index = build_dense_index(chunks)
        indices[domain] = {"chunks": chunks, "bm25": bm25_index, "dense": dense_index}

    gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    process_freeform(gen_client, judge_client, DATASETS_AQA, "AQA", indices)
    process_freeform(gen_client, judge_client, DATASETS_EQA, "EQA", indices)
    process_mcqa(gen_client, DATASETS_MCQA, indices)

    print("[Stage-1] All experiments finished.", flush=True)


if __name__ == "__main__":
    main()
