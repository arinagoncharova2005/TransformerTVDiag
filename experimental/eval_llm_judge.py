"""Step E.3 — independent LLM-as-judge rating of the generated explanations.

For each Step D sample, sends the evidence (the original prompt) plus the
generated explanation to a judge model and asks for a rubric score:
  faithfulness   1-5  (does the explanation only use the given evidence?)
  completeness   1-5  (does it use the most decisive evidence?)
  actionability  1-5  (would it help an SRE act?)
  verdict_correct yes/no (is the support/contradict/unclear call justified?)

Reuses call_llm / extract_json from explain_evidence.py (same endpoint).

Validation: Spearman correlation between the judge's faithfulness score and the
objective per-sample hallucination rate from eval_faithfulness.py. High |corr|
means the judge tracks the objective metric; this is reported with the usual
caveat that judge and explainer may share model-family biases.

Env: LLM_BASE_URL, LLM_TOKEN, JUDGE_MODEL (default: falls back to LLM_MODEL,
     then qwen3-235b-a22b).

Writes:
  logs/gaia/explanation_eval/<config>/judge_per_sample.csv
  logs/gaia/explanation_eval/<config>/judge_report.txt
  logs/gaia/explanation_eval/<config>/judge_raw/sample_<idx>.json

Run (from TransTVDiag_attention_analysis/):
    export LLM_BASE_URL=...; export LLM_TOKEN=...
    python experimental/eval_llm_judge.py --config cb_ce_awl --limit 5   # smoke
    python experimental/eval_llm_judge.py --config cb_ce_awl --resume    # full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import math
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

from explain_evidence import call_llm, extract_json


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of automatically generated diagnoses for microservice
failures on the GAIA dataset. You are shown (1) the EVIDENCE that a diagnosis
system was given and (2) the EXPLANATION it produced (a support/contradict/unclear
verdict on a predicted root cause, a dominant modality, an evidence summary, and a
short explanation).

Rate the explanation on a 1-5 integer scale for each criterion:
- faithfulness: does the explanation reference ONLY services and events present in
  the evidence (5 = no invented facts; 1 = mostly invented)?
- completeness: does it cite the most decisive available evidence (5 = uses the
  strongest signals; 1 = ignores them)?
- actionability: would it help an on-call SRE decide what to do (5 = clearly; 1 = no)?
And judge whether the support/contradict/unclear verdict is justified by the
evidence (verdict_correct: "yes" or "no").

Do not produce any reasoning or thinking blocks. Output ONLY a single JSON object
with exactly these keys, in this order:
{
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "actionability": <1-5>,
  "verdict_correct": "yes" | "no",
  "comment": "<one short sentence>"
}
No extra text before or after the JSON. No markdown code fences."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl")
    p.add_argument("--explanations_root", default="logs/gaia/explanations")
    p.add_argument("--eval_root", default="logs/gaia/explanation_eval")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--max_tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rate_limit_sleep", type=float, default=1.0)
    return p.parse_args()


def build_judge_user_prompt(sample: dict) -> str:
    p = sample["parsed"]
    return (
        "EVIDENCE the system was given:\n"
        f"{sample['prompt']}\n\n"
        "EXPLANATION the system produced:\n"
        f"  rcl_agreement: {p.get('rcl_agreement')}\n"
        f"  dominant_modality: {p.get('dominant_modality')}\n"
        f"  evidence_summary: {p.get('evidence_summary')}\n"
        f"  explanation: {p.get('explanation')}\n"
    )


def to_int(v) -> int | None:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None
def spearman_with_p(a, b) -> tuple[float, float]:
    """Spearman rho + two-sided p. Uses scipy if available, else a t/normal
    approximation (good for the n here) so it also runs on a scipy-free server."""
    try:
        from scipy import stats
        rho, p = stats.spearmanr(a, b)
        return float(rho), float(p)
    except ImportError:
        rho = float(a.corr(b, method="spearman"))
        n = len(a)
        if n <= 2 or abs(rho) >= 1:
            return rho, float("nan")
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        return rho, math.erfc(abs(t) / math.sqrt(2))  # two-sided, normal approx

def extract_judge_json(text: str) -> dict | None:
    """Robust JSON extraction for reasoning judges (e.g. gpt-oss).

    The answer may follow a reasoning preamble or harmony tags, and reasoning
    text can contain stray braces that defeat a naive first-brace/last-brace
    parse. Strip common wrappers, then scan for balanced {...} objects and return
    the last one that parses and carries the expected rubric keys.
    """
    if not text:
        return None
    direct = extract_json(text)
    if isinstance(direct, dict) and "faithfulness" in direct:
        return direct
    cleaned = re.sub(r"<\|[^|]*\|>", " ", text)            # harmony channel tags
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.I)  # <think> blocks
    objs, stack = [], []
    for i, ch in enumerate(cleaned):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            objs.append(cleaned[stack.pop():i + 1])
    for cand in reversed(objs):                            # prefer the final object
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("faithfulness" in obj or "verdict_correct" in obj):
            return obj
    return direct


def main() -> None:
    args = parse_args()
    base_url = os.environ.get("LLM_BASE_URL")
    token = os.environ.get("LLM_TOKEN")
    model = 'gpt-oss-120b'
    # model = os.environ.get("JUDGE_MODEL") or os.environ.get("LLM_MODEL") or "qwen3-235b-a22b"
    if not base_url or not token:
        sys.exit("ERROR: set LLM_BASE_URL and LLM_TOKEN env vars")

    expl_dir = Path(args.explanations_root) / args.config
    out_dir = Path(args.eval_root) / args.config
    raw_dir = out_dir / "judge_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = sorted(expl_dir.glob("sample_*.json"))
    if not sample_paths:
        sys.exit(f"no sample_*.json found in {expl_dir}")

    headers = {"Authorization": f"Bearer {token}"}
    client = httpx.Client(base_url=base_url, headers=headers,
                          timeout=httpx.Timeout(60.0, connect=10.0))

    rows, seen = [], set()
    for path in sample_paths:
        with open(path) as f:
            sample = json.load(f)
        if not sample.get("parsed"):
            continue
        gi = sample["gaia_index"]
        if gi in seen:
            continue
        seen.add(gi)
        if args.limit is not None and len(rows) >= args.limit:
            break

        raw_path = raw_dir / f"sample_{gi}.json"
        if args.resume and raw_path.exists():
            with open(raw_path) as f:
                cached = json.load(f)
            rows.append(cached["row"])
            print(f"gaia={gi} (cached)")
            continue

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(sample)},
        ]
        print(f"gaia={gi} group={sample.get('group')} ...", end=" ", flush=True)
        raw, latency, status = call_llm(client, model, messages,
                                        args.max_tokens, args.temperature, args.retries)
        parsed = extract_judge_json(raw) if status == "ok" else None
        row = {
            "gaia_index": gi,
            "group": sample.get("group"),
            "rcl_correct": bool(sample.get("rcl_correct")),
            "rcl_agreement": sample["parsed"].get("rcl_agreement"),
            "judge_faithfulness": to_int(parsed.get("faithfulness")) if parsed else None,
            "judge_completeness": to_int(parsed.get("completeness")) if parsed else None,
            "judge_actionability": to_int(parsed.get("actionability")) if parsed else None,
            "judge_verdict_correct": (parsed.get("verdict_correct") if parsed else None),
            "status": status,
        }
        print(f"{status}"
              + (f" → faith={row['judge_faithfulness']}" if parsed else ""))
        with open(raw_path, "w") as f:
            json.dump({"row": row, "raw_response": raw, "model": model}, f,
                      indent=2, ensure_ascii=False)
        rows.append(row)
        time.sleep(args.rate_limit_sleep)

    if not rows:
        sys.exit("no samples judged")

    df = pd.DataFrame(rows)
    per_sample_path = out_dir / "judge_per_sample.csv"
    df.to_csv(per_sample_path, index=False)

    lines = ["=== LLM-as-judge report ===",
             f"config: {args.config}   judge model: {model}",
             f"samples judged: {len(df)}  "
             f"(ok: {(df['status'] == 'ok').sum()})", ""]
    for col in ("judge_faithfulness", "judge_completeness", "judge_actionability"):
        s = df[col].dropna()
        if len(s):
            lines.append(f"  {col:<22} mean={s.mean():.2f}  median={s.median():.0f}  "
                         f"min={int(s.min())} max={int(s.max())}")
    if "judge_verdict_correct" in df:
        yes = (df["judge_verdict_correct"].astype(str).str.lower() == "yes").mean()
        lines.append(f"  verdict_correct (yes): {yes:.3f}")
    lines.append("")

    # ----- validation: does the judge agree with the objective check? -----
    # Objective unfaithfulness = any atom / relational / modality error per sample.
    # Atom is ~always 0, so the signal comes from relational/modality; this is the
    # real test of whether the judge catches the structural hallucinations.
    faith_csv = out_dir / "faithfulness_per_sample.csv"
    if faith_csv.exists():
        fdf = pd.read_csv(faith_csv)
        bad_cols = [c for c in ["bad_mentions", "relational_bad", "modality_bad"]
                    if c in fdf.columns]
        fdf["obj_bad"] = fdf[bad_cols].sum(axis=1)
        fdf["obj_unfaithful"] = fdf["obj_bad"] > 0
        merged = df.merge(fdf[["gaia_index", "obj_bad", "obj_unfaithful"]],
                          on="gaia_index").dropna(subset=["judge_faithfulness"])
        lines.append("--- validation vs objective faithfulness (atom+relational+modality) ---")
        if len(merged) >= 3:
            clean = merged[~merged["obj_unfaithful"]]
            flagged = merged[merged["obj_unfaithful"]]
            lines.append(f"  judge faithfulness, objectively clean:   "
                         f"mean={clean['judge_faithfulness'].mean():.2f} (n={len(clean)})")
            if len(flagged):
                lines.append(f"  judge faithfulness, objectively flagged: "
                             f"mean={flagged['judge_faithfulness'].mean():.2f} (n={len(flagged)})")
                lines.append("  (a good judge scores the flagged group lower)")
            if merged["obj_bad"].nunique() > 1 and merged["judge_faithfulness"].nunique() > 1:
                rho, pval = spearman_with_p(merged["judge_faithfulness"], merged["obj_bad"])
                sig = "significant" if pval < 0.05 else "n.s."
                lines.append(f"  Spearman(judge_faithfulness, #objective errors) "
                             f"= {rho:.3f}  (p={pval:.4f}, {sig}; n={len(merged)}; "
                             f"expected negative)")
        else:
            lines.append("  insufficient overlap to compare")
        lines.append("")
    else:
        lines.append("--- validation: run eval_faithfulness.py first to enable ---")
        lines.append("")

    lines.append("caveat: judge and explainer may share model-family biases; "
                 "treat scores as indicative, not ground truth.")

    report = "\n".join(lines)
    (out_dir / "judge_report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\nwrote {per_sample_path}")


if __name__ == "__main__":
    main()
