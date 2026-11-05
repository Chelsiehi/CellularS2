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
DATASETS_MULTIUNIT = [
    DATA_ROOT / "4G_NAS_stage3-Multi-Unit.json",
    DATA_ROOT / "5G_NAS_stage3-Multi-Unit.json",
    DATA_ROOT / "5G_RRC_stage3-Multi-Unit.json",
]
DATASETS_SECURITY = [
    DATA_ROOT / "4G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_NAS_stage3-SecurityReasoning.json",
    DATA_ROOT / "5G_RRC_stage3-SecurityReasoning.json",
]
KG_DIRS = {
    "4G_NAS": Path("4G_NAS_KG_JSON"),
    "5G_NAS": Path("5G_NAS_KG_JSON"),
    "5G_RRC": Path("5G_RRC_KG_JSON"),
}
OUTPUT_ROOT = Path("stage3-multiunit-securityreasoning-specrag-kg")
OUTPUT_MULTI = OUTPUT_ROOT / "multi_unit.json"
OUTPUT_SECURITY = OUTPUT_ROOT / "security_reasoning.json"
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

ALLOWED_MULTI = [
    "Integrity Bypass",
    "Ciphering Bypass / Null Encryption",
    "Identity Exposure / Privacy Leakage",
    "Identification Request Surface",
    "Fake Reject / Unauthorized Deregistration / DoS",
    "Replay / Message Injection",
    "Signalling DoS / Resource Exhaustion",
    "Rate-Limit Missing",
    "Anomaly Detection Missing",
    "Emergency-Exception Abuse",
    "Downgrade Attack",
    "Operator Policy / Subscription Exposure",
    "Identity Procedure Abuse",
]


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


def normalize_label(label: str) -> str:
    if not label:
        return ""
    l = label.strip()
    if l in ALLOWED_MULTI:
        return l
    mp = {
        "fake reject / dos": "Fake Reject / Unauthorized Deregistration / DoS",
        "identity exposure": "Identity Exposure / Privacy Leakage",
        "replay attack": "Replay / Message Injection",
        "downgrade": "Downgrade Attack",
    }
    return mp.get(l.lower(), l)


def normalize_text(text: str) -> List[str]:
    return [t for t in re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split() if t]


def f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def compute_binary_f1(rs: List[Dict[str, object]]) -> float:
    tp = fp = fn = 0
    for r in rs:
        g = str(r["reference_binary"]).lower()
        p = str(r["model_label_binary"]).lower()
        if p == "vulnerable" and g == "vulnerable":
            tp += 1
        elif p == "vulnerable" and g != "vulnerable":
            fp += 1
        elif p != "vulnerable" and g == "vulnerable":
            fn += 1
    return f1(tp, fp, fn)


def compute_multi_f1(rs: List[Dict[str, object]]) -> Tuple[float, float]:
    tp_sum = fp_sum = fn_sum = 0
    per = []
    for label in ALLOWED_MULTI:
        tp = fp = fn = 0
        for r in rs:
            g = {normalize_label(x).lower() for x in r["reference_multi"] if normalize_label(x)}
            p = {normalize_label(x).lower() for x in r["model_label_multi"] if normalize_label(x)}
            hg, hp = label.lower() in g, label.lower() in p
            if hp and hg:
                tp += 1
            elif hp and (not hg):
                fp += 1
            elif (not hp) and hg:
                fn += 1
        per.append(f1(tp, fp, fn))
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    return f1(tp_sum, fp_sum, fn_sum), (sum(per) / len(per) if per else 0.0)


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
    chunks = []
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
        text = "\n".join(parts).strip()
        if not text:
            continue
        chunks.append({"clause_id": cid, "clause_title": title, "text": text, "aug_text": f"{cid} {title} {text}".strip()})
    return chunks


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
            s += idx["idf"][t] * (f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * dl / idx["avgdl"]))) * q
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


def format_contexts(ctx: List[Dict]) -> str:
    return "\n\n-----\n\n".join(f"[{i}] Clause {c.get('clause_id','')} - {c.get('clause_title','')}\n{c.get('text','')}" for i, c in enumerate(ctx, 1))


def pad_contexts(ctx: List[Dict], k: int) -> List[Dict]:
    if len(ctx) >= k:
        return ctx[:k]
    return ctx + [{"clause_id": "", "clause_title": "", "text": "", "aug_text": ""}] * (k - len(ctx))


def build_evidence_from_context(ctx: List[Dict], domain: str) -> str:
    if not ctx:
        return ""
    c = ctx[0]
    cid = str(c.get("clause_id", "")).strip()
    title = str(c.get("clause_title", "")).strip()
    prefix = {"4G_NAS": "TS 24.301", "5G_NAS": "TS 24.501", "5G_RRC": "TS 38.331"}.get(domain, "TS")
    if cid and title:
        return f"{prefix}, Clause {cid}, {title}"
    if cid:
        return f"{prefix}, Clause {cid}"
    return prefix


def context_has_gold_clause(ctx: List[Dict], gold: str | None) -> bool:
    if not gold:
        return False
    g = gold.lower()
    for c in ctx:
        cid = str(c.get("clause_id", "")).lower()
        title = str(c.get("clause_title", "")).lower()
        if g in cid or cid in g or g in title:
            return True
    return False


def run_multiunit(gen: OpenAI, datasets: List[Path], idxs: Dict[str, Dict]) -> Dict[str, object]:
    rs = []
    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        domain = detect_domain(path)
        idx = idxs[domain]
        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[Multi-Unit] {path.name} {i}/{len(data)}", flush=True)
            q = item.get("input", "")
            ctx = hybrid_retrieve(q, idx["chunks"], idx["bm25"], idx["dense"], TOP_K_CONTEXT)
            prompt = (
                f"{domain_prompt(domain)}\n"
                "You are a 4G/5G NAS/RRC specification security analyst.\n"
                "Return ONLY JSON with keys 'label_binary' and 'label_multi'.\n"
                "Binary: Vulnerable or non-Vulnerable.\n"
                "Multi-label from this list:\n" + "\n".join(f"- {x}" for x in ALLOWED_MULTI) + "\n\n"
                f"Context:\n{format_contexts(ctx)}\n\nInput:\n{q}"
            )
            raw = call_llm(gen, MODEL_NAME, [{"role": "user", "content": prompt}], 0.1, 512)
            raw2 = re.sub(r"```json|```", "", raw).strip()
            pb = ""
            pm: List[str] = []
            try:
                j = json.loads(raw2)
                pb = str(j.get("label_binary", "")).strip()
                jl = j.get("label_multi", [])
                if isinstance(jl, list):
                    pm = [normalize_label(str(x)) for x in jl if normalize_label(str(x))]
                elif isinstance(jl, str) and normalize_label(jl):
                    pm = [normalize_label(jl)]
            except Exception:
                pass
            rs.append({"dataset": path.name, "id": item.get("id"), "question": q, "reference_binary": item.get("label_binary", ""), "reference_multi": item.get("label_multi", []) or [], "model_answer_raw": raw, "model_label_binary": pb, "model_label_multi": pm, "retrieved_contexts": pad_contexts(ctx, TOP_K_CONTEXT)})

    b = compute_binary_f1(rs)
    mi, ma = compute_multi_f1(rs)
    return {"type": "Multi-Unit", "count": len(rs), "label_binary_f1": b, "label_multi_micro_f1": mi, "label_multi_macro_f1": ma, "results": rs}


def judge_answer(judge: OpenAI, ref: str, cand: str) -> Tuple[int, str]:
    p = (
        "Grade this security reasoning answer.\n\n"
        "Score 2 = Correct (captures vulnerability rationale, no contradiction).\n"
        "Score 1 = Partially Correct (direction right but missing key points).\n"
        "Score 0 = Incorrect (main reason wrong or conflicts with spec).\n\n"
        f"Reference:\n{ref}\n\nModel answer:\n{cand}\n\nReturn ONLY strict JSON with keys: score, justification."
    )
    raw = call_llm(judge, JUDGE_MODEL, [{"role": "user", "content": p}], 0.0, 256)
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


def run_security(gen: OpenAI, judge: OpenAI, datasets: List[Path], idxs: Dict[str, Dict]) -> Dict[str, object]:
    rs = []
    total_n = score_sum = ev_ok = ev_n = 0
    sd = {"0": 0, "1": 0, "2": 0}
    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_n += len(data)
        domain = detect_domain(path)
        idx = idxs[domain]
        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[Security] {path.name} {i}/{len(data)}", flush=True)
            vuln = item.get("input", "")
            gold_ev = item.get("evidence", "")
            ctx = hybrid_retrieve(vuln, idx["chunks"], idx["bm25"], idx["dense"], TOP_K_CONTEXT)
            prompt = f"{domain_prompt(domain)}\nDescribe the vulnerability and rationale using context.\n\nContext:\n{format_contexts(ctx)}\n\nInput vulnerability:\n{vuln}\n\nAnswer:"
            ans = call_llm(gen, MODEL_NAME, [{"role": "user", "content": prompt}], 0.2, 512)
            s, just = judge_answer(judge, vuln, ans)
            ev = build_evidence_from_context(ctx, domain)
            ok = False
            if gold_ev:
                ev_n += 1
                ok = (ev == gold_ev) or context_has_gold_clause(ctx, gold_ev)
                ev_ok += int(ok)
            score_sum += s
            sd[str(s)] = sd.get(str(s), 0) + 1
            rs.append({"dataset": path.name, "id": item.get("id"), "input": vuln, "reference": vuln, "gold_evidence": gold_ev, "model_answer": ans, "evidence": ev, "retrieved_contexts": pad_contexts(ctx, TOP_K_CONTEXT), "score": s, "evidence_correct": ok, "justification": just})

    return {"type": "SecurityReasoning", "count": total_n, "avg_score": score_sum / total_n if total_n else 0.0, "evidence_accuracy": ev_ok / ev_n if ev_n else 0.0, "evidence_total": ev_n, "score_distribution": sd, "results": rs}


def main() -> None:
    global API_KEY, BASE_URL, JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL, MODEL_NAME
    global OUTPUT_ROOT, OUTPUT_MULTI, OUTPUT_SECURITY, TOP_K_CONTEXT, HYBRID_ALPHA, GRADING_PASSES, LOG_EVERY
    ap = argparse.ArgumentParser(description="Stage3 Multi-Unit + SecurityReasoning SpecRAG+KG (release-safe)")
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
    OUTPUT_MULTI = OUTPUT_ROOT / "multi_unit.json"
    OUTPUT_SECURITY = OUTPUT_ROOT / "security_reasoning.json"
    TOP_K_CONTEXT = max(1, args.top_k_context)
    HYBRID_ALPHA = max(0.0, min(1.0, args.hybrid_alpha))
    GRADING_PASSES = max(1, args.grading_passes)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS_MULTIUNIT + DATASETS_SECURITY:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")
    for d, p in KG_DIRS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing KG dir for {d}: {p}")

    idxs: Dict[str, Dict] = {}
    for domain, kg in KG_DIRS.items():
        chunks = load_chunks(kg)
        idxs[domain] = {"chunks": chunks, "bm25": build_bm25(chunks), "dense": build_dense_index(chunks)}

    gen = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    judge = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    out_m = run_multiunit(gen, DATASETS_MULTIUNIT, idxs)
    out_s = run_security(gen, judge, DATASETS_SECURITY, idxs)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MULTI.write_text(json.dumps(out_m, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_SECURITY.write_text(json.dumps(out_s, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
