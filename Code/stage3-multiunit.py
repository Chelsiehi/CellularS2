from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

DATA_ROOT = Path("")
DATASETS = [
    DATA_ROOT / "4G_NAS_stage3-Multi-Unit.json",
    DATA_ROOT / "5G_NAS_stage3-Multi-Unit.json",
    DATA_ROOT / "5G_RRC_stage3-Multi-Unit.json",
]
OUTPUT_FILE = Path("stage3-multiunit.json")

# Choose your settings directly here:
# API_KEY = "your_generation_api_key"
# BASE_URL = "https://your-openai-compatible-endpoint/v1"
# MODEL_NAME = "your_generation_model"
API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""
LOG_EVERY = 20

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
    "Ciphering Bypass",
    "DoS",
    "Error-Code Oracle",
    "IMSI/IMEI Exposure",
    "Implementation-Dependent Behavior",
    "Linkability / Tracking",
    "Plaintext Identifier Leakage",
    "Plaintext Identifier/Message Exposure",
    "Privacy / Tracking",
    "Privacy/Tracking",
    "Replay Protection Underspecified",
    "Resource Exhaustion",
    "Spoofing / False Paging",
    "Timer Manipulation / Backoff Abuse",
    "Unauthenticated Message Acceptance",
    "Unauthenticated RRC Message",
    "Unencrypted Broadcast",
]


def domain_prompt(path: Path) -> str:
    n = path.name
    if n.startswith("4G_NAS"):
        return "You are a 4G NAS expert (3GPP TS 24.301)."
    if n.startswith("5G_NAS"):
        return "You are a 5G NAS expert (3GPP TS 24.501)."
    if n.startswith("5G_RRC"):
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


def f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def binary_f1(rs: List[Dict]) -> float:
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


def multi_f1(rs: List[Dict]) -> Tuple[float, float]:
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
            elif hp and not hg:
                fp += 1
            elif (not hp) and hg:
                fn += 1
        per.append(f1(tp, fp, fn))
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    return f1(tp_sum, fp_sum, fn_sum), (sum(per) / len(per) if per else 0.0)


def main() -> None:
    global API_KEY, BASE_URL, MODEL_NAME, OUTPUT_FILE, LOG_EVERY
    ap = argparse.ArgumentParser(description="Stage3 Multi-Unit (release-safe)")
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--output-file", default=str(OUTPUT_FILE))
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args()

    if not args.api_key or not args.base_url or not args.model_name:
        raise ValueError("Set API_KEY/BASE_URL/MODEL_NAME.")

    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    OUTPUT_FILE = Path(args.output_file)
    LOG_EVERY = max(1, args.log_every)

    for p in DATASETS:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    all_results = []

    for path in DATASETS:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for i, item in enumerate(data, 1):
            if i % LOG_EVERY == 0 or i == len(data):
                print(f"[Multi-Unit] {path.name} {i}/{len(data)}", flush=True)
            q = item.get("input", "")
            prompt = (
                "You are a 4G/5G NAS/RRC specification security analyst.\n"
                "Return ONLY JSON with keys 'label_binary' and 'label_multi'.\n"
                "Binary: Vulnerable or non-Vulnerable.\n"
                "Multi-label from this list:\n" + "\n".join(f"- {x}" for x in ALLOWED_MULTI) + "\n\n"
                f"Context role:\n{domain_prompt(path)}\n\nInput:\n{q}"
            )
            raw = (client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1).choices[0].message.content or "").strip()
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
            out.append(
                {
                    "dataset": path.name,
                    "id": item.get("id"),
                    "question": q,
                    "reference_binary": item.get("label_binary", ""),
                    "reference_multi": item.get("label_multi", []) or [],
                    "model_label_binary": pb,
                    "model_label_multi": pm,
                }
            )

        all_results.extend(out)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(
            {"type": "Multi-Unit", "results": all_results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
