from __future__ import annotations

import argparse, json, math, random, re, time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_ROOT = Path("")
DATASETS = [DATA_ROOT / "4G_NAS_stage3-TFQA.json", DATA_ROOT / "5G_NAS_stage3-TFQA.json", DATA_ROOT / "5G_RRC_stage3-TFQA.json"]
CHUNK_FILES = {"4G_NAS": Path("4Gnas_paragraph_chunks.jsonl"), "5G_NAS": Path("5Gnas_paragraph_chunks.jsonl"), "5G_RRC": Path("5Grrc_paragraph_chunks.jsonl")}
RAG_FILES = {"4G_NAS": Path("4Gnas_paragraph_rag_chunks.jsonl"), "5G_NAS": Path("5Gnas_paragraph_rag_chunks.jsonl"), "5G_RRC": Path("5Grrc_paragraph_rag_chunks.jsonl")}
COMPONENT_DIRS = {
    "4G_NAS": {"table": Path("SpecFusion/4G_NAS_Table"), "figure": Path("SpecFusion/4G_NAS_Figure")},
    "5G_NAS": {"table": Path("SpecFusion/5G_NAS_Table"), "figure": Path("SpecFusion/5G_NAS_Figure")},
    "5G_RRC": {"table": Path("SpecFusion/5G_RRC_Table"), "figure": Path("SpecFusion/5G_RRC_Figure")},
}
OUTPUT_FILE = Path("stage3-specfusion-specanchor-tfqa.json")
TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY = 8, 0.7, 1, 20
MAX_RETRIES, RETRY_BASE_SLEEP, RETRY_JITTER = 6, 1.2, 0.5
API_KEY = BASE_URL = MODEL_NAME = ""
JUDGE_API_KEY = JUDGE_BASE_URL = JUDGE_MODEL = ""


def normalize_text(text: str) -> List[str]: return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split()
def detect_domain(path: Path) -> str:
    for domain in CHUNK_FILES:
        if path.name.startswith(domain): return domain
    raise ValueError(f"Unknown dataset domain: {path}")
def domain_prompt(domain: str) -> str: return {"4G_NAS": "You are a 4G NAS expert (3GPP TS 24.301).", "5G_NAS": "You are a 5G NAS expert (3GPP TS 24.501).", "5G_RRC": "You are a 5G RRC expert (3GPP TS 38.331)."}[domain]
def evidence_prefix(domain: str) -> str: return {"4G_NAS": "TS 24.301", "5G_NAS": "TS 24.501", "5G_RRC": "TS 38.331"}[domain]


def call_llm(client: OpenAI, model: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return (response.choices[0].message.content or "").strip()
        except Exception as error:
            last_error = error; time.sleep(RETRY_BASE_SLEEP * (2**attempt) + random.random() * RETRY_JITTER)
    return f"[ERROR] {last_error!r}"


def load_paragraphs(path: Path) -> List[Dict]:
    chunks, seen = [], set()
    if not path.exists(): return chunks
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try: obj = json.loads(line)
            except Exception: continue
            clause_id, title, text = str(obj.get("clause_id", "")).strip(), str(obj.get("clause_title", "")).strip(), str(obj.get("text", "")).strip()
            label = f"Clause {clause_id}" if clause_id else ""
            index_text = " ".join(value for value in (label, title, text) if value)
            if not text or index_text in seen: continue
            seen.add(index_text); chunks.append({"component_type": "subclause", "citation_label": label, "clause_id": clause_id, "title": title, "text": text, "index_text": index_text})
    return chunks


def load_components(directory: Path, component_type: str) -> List[Dict]:
    chunks = []
    if not directory.exists(): return chunks
    for path in sorted(directory.glob("*.json")):
        try: obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception: continue
        if component_type == "table":
            title = str(obj.get("table_title", "")).strip(); label = title.split(":", 1)[0].strip() if title else path.stem
        else:
            figure_id, title = str(obj.get("figure_id", "")).strip(), str(obj.get("title", "")).strip(); label = f"Figure {figure_id}" if figure_id else path.stem
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":")); index_text = " ".join(value for value in (label, title, text) if value)
        chunks.append({"component_type": component_type, "citation_label": label, "clause_id": "", "title": title, "text": text, "index_text": index_text})
    return chunks


def load_component_space(domain: str) -> List[Dict]:
    chunks = load_paragraphs(RAG_FILES[domain]) or load_paragraphs(CHUNK_FILES[domain])
    for component_type, directory in COMPONENT_DIRS[domain].items(): chunks.extend(load_components(directory, component_type))
    seen, unique = set(), []
    for chunk in chunks:
        if chunk["index_text"] not in seen: seen.add(chunk["index_text"]); unique.append(chunk)
    return unique


def build_bm25(chunks: List[Dict]) -> Dict:
    tokenized = [normalize_text(chunk["index_text"]) for chunk in chunks]; df = Counter(token for tokens in tokenized for token in set(tokens)); count = len(chunks) or 1; avgdl = sum(map(len, tokenized)) / count
    return {"tokenized": tokenized, "idf": {token: math.log(1 + (count - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}, "avgdl": avgdl}


def bm25_score(query: str, index: Dict) -> List[float]:
    query_frequency, scores = Counter(normalize_text(query)), []
    for tokens in index["tokenized"]:
        tf, length, score = Counter(tokens), len(tokens), 0.0
        for term, query_count in query_frequency.items():
            frequency = tf.get(term, 0)
            if not frequency or term not in index["idf"]: continue
            score += index["idf"][term] * frequency * 2.5 / (frequency + 1.5 * (1 - 0.75 + 0.75 * length / index["avgdl"])) * query_count
        scores.append(score)
    return scores


def build_dense(chunks: List[Dict]) -> Dict:
    vectorizer = TfidfVectorizer(max_features=30000); matrix = vectorizer.fit_transform([chunk["index_text"] for chunk in chunks]); maximum = min(matrix.shape) - 1
    if maximum < 1: return {"vectorizer": vectorizer, "svd": None, "embeddings": matrix.toarray()}
    svd = TruncatedSVD(n_components=min(256, maximum), random_state=42); embeddings = svd.fit_transform(matrix); embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return {"vectorizer": vectorizer, "svd": svd, "embeddings": embeddings}


def hybrid_retrieve(query: str, chunks: List[Dict], bm25: Dict, dense: Dict, top_k: int, alpha: float) -> List[Dict]:
    sparse = bm25_score(query, bm25); maximum = max(sparse) if sparse else 0.0; sparse = [score / maximum for score in sparse] if maximum > 0 else [0.0] * len(chunks)
    query_vector = dense["vectorizer"].transform([query]); query_vector = dense["svd"].transform(query_vector) if dense["svd"] is not None else query_vector.toarray(); query_vector = query_vector.reshape(-1); query_vector /= np.linalg.norm(query_vector) + 1e-8
    semantic = [((score + 1.0) / 2.0) for score in (dense["embeddings"] @ query_vector).tolist()]
    ranked = sorted(enumerate(alpha * lexical + (1 - alpha) * meaning for lexical, meaning in zip(sparse, semantic)), key=lambda pair: pair[1], reverse=True)
    return [chunks[position] for position, _ in ranked[:top_k]]


def format_contexts(contexts: List[Dict]) -> str:
    return "\n\n-----\n\n".join(f"[{number}] [{chunk['component_type'].upper()}] {chunk['citation_label']} - {chunk['title']}\n{chunk['text'][:1800]}" for number, chunk in enumerate(contexts, 1))
def build_evidence(contexts: List[Dict], domain: str) -> str:
    for chunk in contexts:
        if chunk["citation_label"]: return f"{evidence_prefix(domain)}, {chunk['citation_label']}" + (f", {chunk['title']}" if chunk["title"] else "")
    return ""
def evidence_in_top_k(contexts: List[Dict], gold: str) -> bool:
    if not gold: return False
    normalized = re.sub(r"\s+", " ", gold.lower()).strip()
    return any(label and (label in normalized or normalized in label) for chunk in contexts for label in [str(chunk.get("citation_label", "")).lower().strip(), str(chunk.get("clause_id", "")).lower().strip()])


def generate_answer(client: OpenAI, model: str, question: str, contexts: List[Dict], domain: str) -> str:
    prompt = f"{domain_prompt(domain)}\nAnswer the figure or table question using the cited retrieved evidence.\n\nRetrieved Evidence:\n{format_contexts(contexts)}\n\nQuestion:\n{question}\n\nAnswer:"
    return call_llm(client, model, [{"role": "user", "content": prompt}], 0.2, 768)
def judge_answer(client: OpenAI, model: str, question: str, reference: str, answer: str, domain: str) -> int:
    prompt = f"Grade this answer against the reference. Score 2 = correct, 1 = partially correct, 0 = incorrect. Return strict JSON with keys score and justification.\n\nQuestion:\n{question}\n\nReference:\n{reference}\n\nModel answer:\n{answer}"
    raw = call_llm(client, model, [{"role": "system", "content": domain_prompt(domain)}, {"role": "user", "content": prompt}], 0.0, 256)
    try: score = int(json.loads(re.sub(r"```json|```", "", raw).strip()).get("score", 0))
    except Exception:
        match = re.search(r"\b([0-2])\b", raw); score = int(match.group(1)) if match else 0
    return max(0, min(score, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 TFQA with SpecAnchor hybrid RAG")
    parser.add_argument("--api-key", default=API_KEY); parser.add_argument("--base-url", default=BASE_URL); parser.add_argument("--model-name", default=MODEL_NAME); parser.add_argument("--judge-api-key", default=JUDGE_API_KEY); parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL); parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output-file", default=str(OUTPUT_FILE)); parser.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT); parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA); parser.add_argument("--grading-passes", type=int, default=GRADING_PASSES); parser.add_argument("--log-every", type=int, default=LOG_EVERY); args = parser.parse_args()
    if not args.api_key or not args.base_url or not args.model_name: raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME.")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model: raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL.")
    for path in DATASETS + list(RAG_FILES.values()):
        if not path.exists(): raise FileNotFoundError(path)
    generator, judge, indices = OpenAI(api_key=args.api_key, base_url=args.base_url), OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url), {}
    for domain in CHUNK_FILES:
        chunks = load_component_space(domain); indices[domain] = {"chunks": chunks, "bm25": build_bm25(chunks), "dense": build_dense(chunks)}
    results = []
    for path in DATASETS:
        data, domain = json.loads(path.read_text(encoding="utf-8")), detect_domain(path)
        for position, item in enumerate(data, 1):
            if position % max(1, args.log_every) == 0 or position == len(data): print(f"[TFQA] {path.name} {position}/{len(data)}", flush=True)
            question, reference, gold = item["question"], item["answer"], item.get("evidence", ""); index = indices[domain]
            contexts = hybrid_retrieve(question, index["chunks"], index["bm25"], index["dense"], max(1, args.top_k_context), min(1.0, max(0.0, args.hybrid_alpha)))
            answer, evidence = generate_answer(generator, args.model_name, question, contexts, domain), build_evidence(contexts, domain)
            scores = [judge_answer(judge, args.judge_model, question, reference, answer, domain) for _ in range(max(1, args.grading_passes))]
            results.append({"id": item.get("id"), "question": question, "reference": reference, "gold_evidence": gold, "model_answer": answer, "evidence": evidence, "score": max(set(scores), key=scores.count), "evidence_correct": evidence_in_top_k(contexts, gold)})
    output = Path(args.output_file); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps({"type": "TFQA", "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__": main()
