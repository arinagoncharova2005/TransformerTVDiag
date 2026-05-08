"""Step D — generate LLM explanations from evidence summary.

Reads:
  - logs/gaia/evidence/<config>/sanity_per_sample.csv  (for sample selection)
  - logs/gaia/evidence/<config>/evidence_fold_{k}.jsonl  (for bundles)

For each selected sample, formats the bundle as a compact prompt, calls
an OpenAI-compatible chat-completions endpoint, and parses a JSON
response of the form:

    {
      "rcl_agreement": "support" | "contradict" | "unclear",
      "dominant_modality": "trace" | "log" | "metric" | "none",
      "evidence_summary": "<1-2 sentences>",
      "explanation": "<1-3 sentences>"
    }

Writes:
  - logs/gaia/explanations/<config>/sample_{gaia_index}.json   (per sample)
  - logs/gaia/explanations/<config>/summary.csv                (aggregate)

Configuration via environment variables:
  LLM_BASE_URL  base URL ending BEFORE /v1, e.g. https://api.example.com
  LLM_TOKEN     bearer token
  LLM_MODEL     model id (default: qwen3-235b-a22b)

Run:
    export LLM_BASE_URL=...; export LLM_TOKEN=...
    python experimental/explain_evidence.py --limit 5      # quick smoke
    python experimental/explain_evidence.py                # full ~200 samples
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd


SYSTEM_PROMPT = """/no_think
You are an SRE assistant analyzing a model-generated diagnosis for a microservice failure on the GAIA dataset (10 service instances: webservice1/2, mobservice1/2, dbservice1/2, redisservice1/2, logservice1/2). A multi-modal Graph Transformer model produced a predicted root-cause service and a failure type. You see the model's top attention signals from three modalities: trace (RPC edges with error types: PD = performance degradation, 500 = server error, 400 = client error), log (templates and raw lines per service), and metric (anomaly counts on hosts).

Your task:
1. Decide whether the evidence supports, contradicts, or is unclear about the model's predicted root cause.
2. Identify which modality provided the most decisive signal.
3. Summarize the evidence (1-2 sentences) citing concrete services or events.
4. Explain (1-3 sentences) why the predicted root cause is or is not consistent with the evidence.

Do not produce any reasoning, planning, or thinking blocks. Output ONLY a single JSON object with exactly these four keys, in this order:
{
  "rcl_agreement": "support" | "contradict" | "unclear",
  "dominant_modality": "trace" | "log" | "metric" | "none",
  "evidence_summary": "<1-2 sentences>",
  "explanation": "<1-3 sentences>"
}
No extra text before or after the JSON. No markdown code fences."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl")
    p.add_argument("--evidence_root", default="logs/gaia/evidence")
    p.add_argument("--out_root", default="logs/gaia/explanations")
    p.add_argument("--limit", type=int, default=None,
                   help="if set, only run on first N selected samples (smoke)")
    p.add_argument("--top_k_per_modality", type=int, default=3,
                   help="how many services per modality to include in prompt")
    p.add_argument("--max_raw_log_chars", type=int, default=160,
                   help="truncate raw log messages to this many chars")
    p.add_argument("--rate_limit_sleep", type=float, default=1.0,
                   help="sleep between API calls (sec)")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42,
                   help="seed for sample selection")
    p.add_argument("--n_per_group", type=int, default=40,
                   help="samples per stratification group (5 groups -> ~200)")
    p.add_argument("--n_random", type=int, default=40,
                   help="random samples on top")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose output file already exists")
    return p.parse_args()


# ---------- Sample selection ------------------------------------------------

def select_samples(sanity_df: pd.DataFrame, n_per_group: int, n_random: int,
                   seed: int) -> pd.DataFrame:
    """Pick a stratified mix for case-study coverage."""
    rng = random.Random(seed)

    def take(df: pd.DataFrame, n: int, sort_by: Optional[str] = None) -> pd.DataFrame:
        if df.empty:
            return df.head(0)
        if sort_by is not None:
            df = df.sort_values(sort_by, ascending=False)
        return df.head(n)

    groups = {
        "A_trace_agree_correct": take(
            sanity_df[(sanity_df["trace_top1_is_rcl_pred"]) &
                      (sanity_df["rcl_correct"])],
            n_per_group, "trace_top1_lift"),
        "B_trace_agree_wrong": take(
            sanity_df[(sanity_df["trace_top1_is_rcl_pred"]) &
                      (~sanity_df["rcl_correct"])],
            n_per_group, "trace_top1_lift"),
        "C_trace_disagree_correct": take(
            sanity_df[(~sanity_df["trace_top1_is_rcl_pred"]) &
                      (sanity_df["rcl_correct"])],
            n_per_group, "trace_top1_lift"),
        "D_log_dominates": take(
            sanity_df[sanity_df["winning_modality"] == "log"],
            n_per_group, "log_top1_lift"),
    }
    selected = pd.concat([
        df.assign(group=name) for name, df in groups.items()
    ])

    remaining = sanity_df[~sanity_df["gaia_index"].isin(selected["gaia_index"])]
    if n_random > 0 and not remaining.empty:
        rand_pick = remaining.sample(
            n=min(n_random, len(remaining)),
            random_state=seed).assign(group="E_random")
        selected = pd.concat([selected, rand_pick])

    return selected.drop_duplicates("gaia_index").reset_index(drop=True)


# bundle index

def build_bundle_index(evidence_dir: Path) -> dict:
    """Read all fold jsonl files once, index by gaia_index."""
    index = {}
    for jsonl_path in sorted(evidence_dir.glob("evidence_fold_*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                bundle = json.loads(line)
                index[int(bundle["gaia_index"])] = bundle
    return index


# prompt formatting

def format_bundle_for_prompt(bundle: dict, top_k: int,
                             max_raw_chars: int) -> str:
    lines = []
    lines.append(f"Sample id (GAIA): {bundle['gaia_index']}")
    lines.append(f"Model prediction: root_cause={bundle['rcl_pred_service']}, "
                 f"failure_type={bundle['fti_pred']}")
    lines.append("")

    for modality in ("trace", "log", "metric"):
        entries = bundle["evidence_by_modality"].get(modality, [])[:top_k]
        if not entries:
            continue
        lines.append(f"--- {modality.upper()} attention top-{len(entries)} ---")
        for e in entries:
            tag = " [predicted RCL]" if e["is_predicted_rcl"] else ""
            lines.append(f"* {e['service']} (lift={e['attention_lift']:.2f}x){tag}")
            details = e["details"]
            if modality == "trace":
                edges = details.get("edges", [])[:3]
                if not edges:
                    lines.append("    (no trace events)")
                for edg in edges:
                    lines.append(
                        f"    {edg['direction']} {edg['peer_service']}: "
                        f"{edg['error_description']} (count={edg['count']})")
            elif modality == "log":
                tpls = details.get("templates", [])[:2]
                if not tpls:
                    lines.append("    (no log events)")
                for tpl in tpls:
                    raw = tpl.get("raw_examples", [])
                    if raw:
                        msg = raw[0]["message"][:max_raw_chars]
                        lines.append(
                            f"    tid={tpl['template_id']} (n={tpl['count']}): {msg}")
                    elif tpl.get("template_text"):
                        lines.append(
                            f"    tid={tpl['template_id']} (n={tpl['count']}): "
                            f"{tpl['template_text']}")
                    else:
                        lines.append(
                            f"    tid={tpl['template_id']} (n={tpl['count']}, no text)")
            elif modality == "metric":
                anoms = details.get("anomalies", [])[:5]
                hosts = details.get("hosts", [])
                if not anoms:
                    lines.append("    (no metric anomalies)")
                else:
                    names = ", ".join(a["metric_name"] for a in anoms)
                    lines.append(f"    hosts={hosts}; anomalies: {names}")
        lines.append("")
    return "\n".join(lines).rstrip()


# LLM call
def call_llm(client: httpx.Client, model: str, messages: list,
             max_tokens: int, temperature: float, retries: int) -> tuple:
    """Returns (raw_text, latency_sec, status). status in {ok, http_error, exception}."""
    delay = 1.0
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
            dt = time.time() - t0
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return content, dt, "ok"
            if resp.status_code == 429:
                ra = delay
                try:
                    body = resp.json()
                    ra = float(body.get("retry_after", delay))
                except Exception:
                    pass
                print(f"    rate-limited, sleeping {ra:.1f}s "
                      f"(attempt {attempt + 1}/{retries})")
                time.sleep(ra)
                delay *= 2
                continue
            print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.text, dt, "http_error"
        except Exception as e:
            print(f"    exception: {e}")
            time.sleep(delay)
            delay *= 2
    return None, 0.0, "exhausted"


# response parsing
def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        end = s.rfind("```")
        if end >= 0:
            s = s[:end].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    args = parse_args()

    base_url = os.environ.get("LLM_BASE_URL")
    token = os.environ.get("LLM_TOKEN")
    model = os.environ.get("LLM_MODEL", "qwen3-235b-a22b")
    if not base_url or not token:
        sys.exit("ERROR: set LLM_BASE_URL and LLM_TOKEN env vars")

    evidence_dir = Path(args.evidence_root) / args.config
    out_dir = Path(args.out_root) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading sanity table from {evidence_dir}/sanity_per_sample.csv")
    sanity = pd.read_csv(evidence_dir / "sanity_per_sample.csv")
    sanity["trace_top1_is_rcl_pred"] = sanity["trace_top1_is_rcl_pred"].astype(bool)
    sanity["rcl_correct"] = sanity["rcl_correct"].astype(bool)
    print(f"  sanity rows: {len(sanity)}")

    selected = select_samples(sanity, args.n_per_group, args.n_random, args.seed)
    print(f"  selected: {len(selected)} samples")
    print(selected["group"].value_counts().to_string())
    if args.limit is not None:
        selected = selected.head(args.limit)
        print(f"  --limit applied: {len(selected)} samples")

    print(f"\nbuilding bundle index from {evidence_dir}/evidence_fold_*.jsonl ...")
    bundle_index = build_bundle_index(evidence_dir)
    print(f"  indexed {len(bundle_index)} bundles")

    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(60.0, connect=10.0)
    client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    summary_rows = []
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        gaia_index = int(row.gaia_index)
        out_path = out_dir / f"sample_{gaia_index}.json"
        if args.resume and out_path.exists():
            print(f"[{i}/{len(selected)}] gaia={gaia_index} group={row.group} "
                  f"(skipped, exists)")
            try:
                with open(out_path) as f:
                    cached = json.load(f)
                summary_rows.append({
                    "gaia_index": gaia_index, "group": row.group,
                    "fold": cached.get("fold"),
                    "rcl_pred_service": cached.get("rcl_pred_service"),
                    "rcl_correct": bool(row.rcl_correct),
                    "rcl_agreement": cached.get("parsed", {}).get("rcl_agreement"),
                    "dominant_modality": cached.get("parsed", {}).get("dominant_modality"),
                    "status": cached.get("status"),
                    "latency_sec": cached.get("latency_sec"),
                })
            except Exception:
                pass
            continue

        bundle = bundle_index.get(gaia_index)
        if bundle is None:
            print(f"[{i}/{len(selected)}] gaia={gaia_index}: bundle missing, skip")
            continue

        prompt = format_bundle_for_prompt(
            bundle, args.top_k_per_modality, args.max_raw_log_chars)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        print(f"[{i}/{len(selected)}] gaia={gaia_index} group={row.group} "
              f"rcl={bundle['rcl_pred_service']} ...", end=" ", flush=True)
        raw, latency, status = call_llm(
            client, model, messages,
            args.max_tokens, args.temperature, args.retries)
        parsed = extract_json(raw) if status == "ok" else None
        print(f"{status} ({latency:.1f}s)"
              + (f" → {parsed.get('rcl_agreement')}" if parsed else ""))

        with open(out_path, "w") as f:
            json.dump({
                "gaia_index": gaia_index,
                "fold": bundle["fold"],
                "group": row.group,
                "rcl_pred": bundle["rcl_pred"],
                "rcl_pred_service": bundle["rcl_pred_service"],
                "fti_pred": bundle["fti_pred"],
                "rcl_correct": bool(row.rcl_correct),
                "fti_correct": bool(row.fti_correct),
                "prompt": prompt,
                "raw_response": raw,
                "parsed": parsed,
                "status": status,
                "latency_sec": latency,
                "model": model,
            }, f, indent=2, ensure_ascii=False)

        summary_rows.append({
            "gaia_index": gaia_index,
            "group": row.group,
            "fold": bundle["fold"],
            "rcl_pred_service": bundle["rcl_pred_service"],
            "rcl_correct": bool(row.rcl_correct),
            "rcl_agreement": parsed.get("rcl_agreement") if parsed else None,
            "dominant_modality": parsed.get("dominant_modality") if parsed else None,
            "status": status,
            "latency_sec": latency,
        })

        time.sleep(args.rate_limit_sleep)

    summary_path = out_dir / "summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nwrote summary: {summary_path} ({len(summary_rows)} rows)")
    else:
        print("\nno samples processed")


if __name__ == "__main__":
    main()
