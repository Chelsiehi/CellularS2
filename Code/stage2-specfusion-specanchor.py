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

OUTPUT_ROOT = Path("stage2-specfusion")

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


def evidence_prefix(domain: str) -> str:
    return {"4G_NAS": "TS 24.301", "5G_NAS": "TS 24.501", "5G_RRC": "TS 38.331"}.get(domain, "TS")


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
            try:
                obj = json.loads(line)
            except Exception:
                continue
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
    merged: List[Dict] = []
    seen = set()
    subclauses = load_chunks(RAG_FILES[domain])
    if not subclauses:
        subclauses = load_chunks(CHUNK_FILES[domain])
    components = list(subclauses)
    for component_type, directory in COMPONENT_DIRS[domain].items():
        components.extend(load_structured_components(directory, component_type))
    for component in components:
        key = component["index_text"].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(component)
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
        raise RuntimeError("scikit-learn is required for SpecAnchor dense retrieval (TF-IDF + SVD).")
    corpus = [c["index_text"] for c in chunks]
    if not corpus:
        return {"vectorizer": None, "svd": None, "embeddings": np.zeros((0, 1))}
    vectorizer = TfidfVectorizer(max_features=30000)
    matrix = vectorizer.fit_transform(corpus)
    max_components = min(matrix.shape[0], matrix.shape[1]) - 1
    if max_components < 1:
        return {"vectorizer": vectorizer, "svd": None, "embeddings": matrix.toarray()}
    svd = TruncatedSVD(n_components=min(256, max_components), random_state=42)
    embeddings = svd.fit_transform(matrix)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return {"vectorizer": vectorizer, "svd": svd, "embeddings": embeddings}


def dense_score(query: str, dense_index: Dict) -> List[float]:
    vectorizer = dense_index["vectorizer"]
    svd = dense_index["svd"]
    embeddings = dense_index["embeddings"]
    if vectorizer is None or embeddings.shape[0] == 0:
        return []
    query_vector = vectorizer.transform([query])
    query_embedding = svd.transform(query_vector) if svd is not None else query_vector.toarray()
    query_embedding = query_embedding.reshape(-1)
    query_embedding /= np.linalg.norm(query_embedding) + 1e-8
    return (embeddings @ query_embedding).tolist()


def hybrid_retrieve(
    question: str,
    chunks: List[Dict],
    bm25_index: Dict,
    dense_index: Dict,
    k: int,
    alpha: float,
) -> List[Dict]:
    if not chunks:
        return []
    sparse_scores = bm25_score(question, bm25_index)
    max_sparse = max(sparse_scores) if sparse_scores else 0.0
    sparse_normalized = (
        [score / max_sparse for score in sparse_scores]
        if max_sparse > 0
        else [0.0] * len(chunks)
    )
    dense_scores = dense_score(question, dense_index)
    dense_normalized = (
        [(score + 1.0) / 2.0 for score in dense_scores]
        if dense_scores
        else [0.0] * len(chunks)
    )
    scores = [
        alpha * sparse + (1.0 - alpha) * dense
        for sparse, dense in zip(sparse_normalized, dense_normalized)
    ]
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return [chunks[i] for i, _ in ranked[:k]]


def format_contexts(contexts: List[Dict]) -> str:
    out = []
    for i, c in enumerate(contexts, 1):
        label = c.get("citation_label", "") or "Unlabeled component"
        title = c.get("clause_title", "")
        heading = f"[{i}] {label}" + (f" - {title}" if title and title not in label else "")
        out.append(f"{heading}\n{c.get('text', '')}")
    return "\n\n-----\n\n".join(out)


def pad_contexts(contexts: List[Dict], k: int) -> List[Dict]:
    if len(contexts) >= k:
        return contexts[:k]
    return contexts + [{"component_type": "", "citation_label": "", "clause_id": "", "clause_title": "", "text": "", "index_text": ""}] * (k - len(contexts))


def build_evidence_from_context(contexts: List[Dict], domain: str) -> str:
    if not contexts:
        return ""
    c = contexts[0]
    label = str(c.get("citation_label", "")).strip()
    title = str(c.get("clause_title", "")).strip()
    if not label:
        return ""
    return f"{evidence_prefix(domain)}, {label}" + (f", {title}" if title and title not in label else "")


def context_has_gold_clause(contexts: List[Dict], gold: str | None) -> bool:
    if not gold:
        return False
    g = re.sub(r"\s+", " ", gold.lower()).strip()
    for c in contexts:
        labels = [
            str(c.get("citation_label", "")).lower().strip(),
            str(c.get("clause_id", "")).lower().strip(),
            str(c.get("clause_title", "")).lower().strip(),
        ]
        # Empty strings must never participate: "" in g is always True.
        if any(label and (g in label or label in g) for label in labels):
            return True
    return False


# ===================== Generation / Judge ===================== #


def generate_answer(gen_client: OpenAI, question: str, contexts: List[Dict], qa_type: str, domain: str) -> str:
    task = (
        "This is an extractive question. Prefer exact text from context."
        if qa_type == "EQA"
        else "This is an analytical question. Infer from context without contradiction."
    )
    prompt = (
        f"{domain_prompt(domain)}\n{task}\n\n"
        f"Context:\n{format_contexts(contexts)}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
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
    results = []
    total_score = 0
    total_n = 0
    ev_correct = 0
    ev_total = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        chunks = indices[domain]["chunks"]
        bm25 = indices[domain]["bm25"]
        dense = indices[domain]["dense"]

        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[{qa_type}] {path.name} {i}/{len(data)}", flush=True)

            q = item["question"]
            ref = item["answer"]
            gold_ev = item.get("evidence")
            ctxs = hybrid_retrieve(q, chunks, bm25, dense, TOP_K_CONTEXT, HYBRID_ALPHA)
            ans = generate_answer(gen_client, q, ctxs, qa_type, domain)
            score, just = grade_multiple(judge_client, q, ref, ans, domain)
            ev = build_evidence_from_context(ctxs, domain)

            ok_ev = False
            if gold_ev:
                ev_total += 1
                ok_ev = (ev == gold_ev) or context_has_gold_clause(ctxs, gold_ev)
                ev_correct += int(ok_ev)

            total_score += score
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
    out = OUTPUT_ROOT / f"{qa_type.lower()}.json"
    out.write_text(
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


def process_mcqa(gen_client: OpenAI, datasets: List[Path], indices: Dict[str, Dict]) -> None:
    results = []
    total_n = 0
    correct = 0
    ev_correct = 0
    ev_total = 0

    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        chunks = indices[domain]["chunks"]
        bm25 = indices[domain]["bm25"]
        dense = indices[domain]["dense"]

        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[MCQA] {path.name} {i}/{len(data)}", flush=True)

            q = item["question"]
            opts = item["options"]
            gold = str(item["answer"]).strip().upper()
            gold_ev = item.get("evidence")

            rq = q + " " + " ".join(str(v) for v in opts.values())
            ctxs = hybrid_retrieve(rq, chunks, bm25, dense, TOP_K_CONTEXT, HYBRID_ALPHA)
            prompt = (
                f"{domain_prompt(domain)}\nUse context to select option. Reply A/B/C/D only.\n\n"
                f"Context:\n{format_contexts(ctxs)}\n\nQ: {q}\n"
                + "\n".join(f"{k}: {v}" for k, v in opts.items())
            )
            raw = call_llm(gen_client, MODEL_NAME, [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=64)
            m = re.search(r"\b([A-D])\b", raw.upper())
            pred = m.group(1) if m else "X"
            ok = pred == gold
            correct += int(ok)

            ev = build_evidence_from_context(ctxs, domain)
            ok_ev = False
            if gold_ev:
                ev_total += 1
                ok_ev = (ev == gold_ev) or context_has_gold_clause(ctxs, gold_ev)
                ev_correct += int(ok_ev)

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
    out = OUTPUT_ROOT / "mcqa.json"
    out.write_text(
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
    global OUTPUT_ROOT, TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY

    parser = argparse.ArgumentParser(description="Stage-2 chunk RAG generation (release-safe).")
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
        raise RuntimeError("scikit-learn is required. Please install: pip install scikit-learn")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_ROOT = Path(args.output_root)
    TOP_K_CONTEXT = max(1, args.top_k_context)
    HYBRID_ALPHA = min(1.0, max(0.0, args.hybrid_alpha))
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS_AQA + DATASETS_EQA + DATASETS_MCQA:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")
    for d, p in CHUNK_FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing chunk file for {d}: {p}")
        if not RAG_FILES[d].exists():
            raise FileNotFoundError(f"Missing rag file for {d}: {RAG_FILES[d]}")

    indices: Dict[str, Dict] = {}
    for domain in CHUNK_FILES:
        c = load_merged_chunks_for_domain(domain)
        indices[domain] = {
            "chunks": c,
            "bm25": build_bm25(c),
            "dense": build_dense_index(c),
        }

    gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    process_mcqa(gen_client, DATASETS_MCQA, indices)
    process_freeform(gen_client, judge_client, DATASETS_AQA, "AQA", indices)
    process_freeform(gen_client, judge_client, DATASETS_EQA, "EQA", indices)


if __name__ == "__main__":
    main()
