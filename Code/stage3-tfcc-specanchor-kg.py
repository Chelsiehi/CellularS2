from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None
    TruncatedSVD = None

DATA_ROOT = Path("")
TFQA_DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-TFQA.json",
    DATA_ROOT / "5G_NAS_stage3-TFQA.json",
    DATA_ROOT / "5G_RRC_stage3-TFQA.json",
]
CCQA_DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_NAS_stage3-CCQA.json",
    DATA_ROOT / "5G_RRC_stage3-CCQA.json",
]
KG_DIRS = {
    "4G_NAS": Path("4G_NAS_JSON"),
    "5G_NAS": Path("5G_NAS_JSON"),
    "5G_RRC": Path("5G_RRC_JSON"),
}
OUTPUT_ROOT = Path("stage3-tfcc-specrag-kg")
OUTPUT_TFQA = OUTPUT_ROOT / "tfqa.json"
OUTPUT_CCQA = OUTPUT_ROOT / "ccqa.json"
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


def detect_domain(path: Path) -> str:
    n = path.name
    if n.startswith("4G_NAS"):
        return "4G_NAS"
    if n.startswith("5G_NAS"):
        return "5G_NAS"
    if n.startswith("5G_RRC"):
        return "5G_RRC"
    raise ValueError(f"Unknown domain for {path}")


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


def normalize_text(text: str) -> List[str]:
    return [t for t in re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split() if t]


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


def load_chunks(dir_path: Path) -> List[Dict]:
    out = []
    for fp in sorted(dir_path.glob("*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = str(obj.get("clause_id", "")).strip()
        title = str(obj.get("title", "")).strip()
        parts: List[str] = []
        if title:
            parts.append(title)
        raw = obj.get("text")
        if isinstance(raw, list):
            parts.extend([str(x) for x in raw if str(x).strip()])
        elif isinstance(raw, str):
            parts.append(raw)
        for x in obj.get("figures", []) or []:
            parts.append(f"Figure: {x}")
        for x in obj.get("tables", []) or []:
            parts.append(f"Table: {x}")
        text = "\n".join(parts).strip()
        if not text:
            continue
        out.append({"clause_id": cid, "clause_title": title, "text": text, "aug_text": f"{cid} {title} {text}".strip()})
    return out


def build_bm25(chunks: List[Dict]) -> Dict:
    tok = [normalize_text(c["aug_text"]) for c in chunks]
    df = Counter()
    for t in tok:
        for w in set(t):
            df[w] += 1
    n = len(chunks)
    avg = sum(len(t) for t in tok) / max(n, 1)
    idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
    return {"tokenized": tok, "idf": idf, "avgdl": avg}


def bm25_score(query: str, idx: Dict) -> List[float]:
    qf = Counter(normalize_text(query))
    out = []
    for toks in idx["tokenized"]:
        dl = len(toks)
        tf = Counter(toks)
        s = 0.0
        for t, q in qf.items():
            if t not in idx["idf"]:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf_t = idx["idf"][t]
            s += idf_t * (f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * dl / idx["avgdl"]))) * q
        out.append(s)
    return out


def build_dense_index(chunks: List[Dict]) -> Optional[Dict]:
    if TfidfVectorizer is None or TruncatedSVD is None:
        return None
    corpus = [c["aug_text"] for c in chunks]
    if not corpus:
        return None
    vec = TfidfVectorizer(max_features=30000)
    X = vec.fit_transform(corpus)
    n_comp = min(256, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z = svd.fit_transform(X)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return {"vectorizer": vec, "svd": svd, "embeddings": Z}


def dense_score(query: str, d: Optional[Dict]) -> List[float]:
    if d is None:
        return []
    q = d["vectorizer"].transform([query])
    z = d["svd"].transform(q)
    u = (z / (np.linalg.norm(z) + 1e-8)).reshape(-1)
    return (d["embeddings"] @ u).tolist()


def hybrid_retrieve(query: str, chunks: List[Dict], b: Dict, d: Optional[Dict], k: int) -> List[Dict]:
    if not chunks:
        return []
    bs = bm25_score(query, b)
    mb = max(bs) if bs else 0.0
    bn = [x / mb for x in bs] if mb > 0 else [0.0] * len(chunks)
    ds = dense_score(query, d)
    dn = [(x + 1.0) / 2.0 for x in ds] if ds else [0.0] * len(chunks)
    hs = [HYBRID_ALPHA * x + (1 - HYBRID_ALPHA) * y for x, y in zip(bn, dn)]
    rk = sorted(enumerate(hs), key=lambda t: t[1], reverse=True)[:k]
    return [chunks[i] for i, _ in rk]


def format_contexts(contexts: List[Dict]) -> str:
    return "\n\n-----\n\n".join(f"[{i}] Clause {c.get('clause_id','')} - {c.get('clause_title','')}\n{c.get('text','')}" for i, c in enumerate(contexts, 1))


def pad_contexts(ctx: List[Dict], k: int) -> List[Dict]:
    if len(ctx) >= k:
        return ctx[:k]
    return ctx + [{"clause_id": "", "clause_title": "", "text": "", "aug_text": ""}] * (k - len(ctx))


def build_evidence_from_context(contexts: List[Dict], domain: str) -> str:
    if not contexts:
        return ""
    c = contexts[0]
    cid = str(c.get("clause_id", "")).strip()
    title = str(c.get("clause_title", "")).strip()
    if cid and title:
        return f"{evidence_prefix(domain)}, Clause {cid}, {title}"
    if cid:
        return f"{evidence_prefix(domain)}, Clause {cid}"
    return evidence_prefix(domain)


def context_has_gold_clause(contexts: List[Dict], gold: str | None) -> bool:
    if not gold:
        return False
    g = gold.lower()
    for c in contexts:
        cid = str(c.get("clause_id", "")).lower()
        title = str(c.get("clause_title", "")).lower()
        if g in cid or cid in g or g in title:
            return True
    return False


def generate_answer(client: OpenAI, q: str, ctx: List[Dict], domain: str) -> str:
    p = f"{domain_prompt(domain)}\nAnswer concisely using context.\n\nContext:\n{format_contexts(ctx)}\n\nQuestion:\n{q}\n\nAnswer:"
    return call_llm(client, MODEL_NAME, [{"role": "user", "content": p}], 0.2, 512)


def judge_answer(client: OpenAI, q: str, ref: str, cand: str, domain: str) -> Tuple[int, str]:
    p = (
        "Grade this answer against the reference.\n\n"
        "Score 2 = Correct (covers core reasons, no obvious contradiction).\n"
        "Score 1 = Partially Correct (direction right but missing key points or includes unreliable extensions).\n"
        "Score 0 = Incorrect (main reason wrong or conflicts with spec).\n\n"
        f"Question:\n{q}\n\nReference answer:\n{ref}\n\nModel answer:\n{cand}\n\nReturn ONLY strict JSON with keys: score, justification."
    )
    raw = call_llm(client, JUDGE_MODEL, [{"role": "system", "content": domain_prompt(domain)}, {"role": "user", "content": p}], 0.0, 256)
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


def grade_multiple(client: OpenAI, q: str, ref: str, ans: str, domain: str) -> Tuple[int, str]:
    scores = []
    just = ""
    for _ in range(max(1, GRADING_PASSES)):
        s, j = judge_answer(client, q, ref, ans, domain)
        scores.append(s)
        just = just or j
    return max(set(scores), key=scores.count), just


def run_task(gen: OpenAI, judge: OpenAI, datasets: List[Path], indices: Dict[str, Dict], task_name: str) -> Dict[str, object]:
    results = []
    total_n = score_sum = ev_ok = ev_n = 0
    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        idx = indices[domain]
        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[{task_name}] {path.name} {i}/{len(data)}", flush=True)
            q = item["question"]
            ref = item["answer"]
            gold_ev = item.get("evidence", "")
            ctx = hybrid_retrieve(q, idx["chunks"], idx["bm25"], idx["dense"], TOP_K_CONTEXT)
            ans = generate_answer(gen, q, ctx, domain)
            s, just = grade_multiple(judge, q, ref, ans, domain)
            ev = build_evidence_from_context(ctx, domain)
            ok = False
            if gold_ev:
                ev_n += 1
                ok = (ev == gold_ev) or context_has_gold_clause(ctx, gold_ev)
                ev_ok += int(ok)
            score_sum += s
            results.append({"dataset": path.name, "id": item.get("id"), "question": q, "reference": ref, "gold_evidence": gold_ev, "model_answer": ans, "evidence": ev, "retrieved_contexts": pad_contexts(ctx, TOP_K_CONTEXT), "score": s, "evidence_correct": ok, "justification": just})
    return {"type": task_name, "count": total_n, "avg_score": score_sum / total_n if total_n else 0.0, "evidence_accuracy": ev_ok / ev_n if ev_n else 0.0, "evidence_total": ev_n, "results": results}


def main() -> None:
    global API_KEY, BASE_URL, JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL, MODEL_NAME
    global OUTPUT_ROOT, OUTPUT_TFQA, OUTPUT_CCQA, TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY
    ap = argparse.ArgumentParser(description="Stage3 TFQA+CCQA SpecRAG+KG (release-safe)")
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    ap.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    ap.add_argument("--top-k-context", type=int, default=TOP_K_CONTEXT)
    ap.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    ap.add_argument("--grading-passes", type=int, default=GRADING_PASSES)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args()

    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set generation API_KEY/BASE_URL/MODEL_NAME.")
    if not args.judge_api_key or not args.judge_base_url or not args.judge_model:
        raise ValueError("Set judge JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL.")
    if TfidfVectorizer is None or TruncatedSVD is None:
        raise RuntimeError("scikit-learn is required. Install: pip install scikit-learn")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    JUDGE_API_KEY = args.judge_api_key
    JUDGE_BASE_URL = args.judge_base_url
    JUDGE_MODEL = args.judge_model
    OUTPUT_ROOT = Path(args.output_root)
    OUTPUT_TFQA = OUTPUT_ROOT / "tfqa.json"
    OUTPUT_CCQA = OUTPUT_ROOT / "ccqa.json"
    TOP_K_CONTEXT = max(1, args.top_k_context)
    HYBRID_ALPHA = max(0.0, min(1.0, args.hybrid_alpha))
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in TFQA_DATASETS + CCQA_DATASETS:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")
    for d, p in KG_DIRS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing KG dir for {d}: {p}")

    indices: Dict[str, Dict] = {}
    for domain, kg_dir in KG_DIRS.items():
        chunks = load_chunks(kg_dir)
        indices[domain] = {"chunks": chunks, "bm25": build_bm25(chunks), "dense": build_dense_index(chunks)}

    gen = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    tfqa_out = run_task(gen, judge, TFQA_DATASETS, indices, "TFQA")
    ccqa_out = run_task(gen, judge, CCQA_DATASETS, indices, "CCQA")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_TFQA.write_text(json.dumps(tfqa_out, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_CCQA.write_text(json.dumps(ccqa_out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
