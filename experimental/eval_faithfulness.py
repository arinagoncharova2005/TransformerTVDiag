"""Step E.1 — objective faithfulness / hallucination check for LLM explanations.

No LLM calls. Reads the Step D outputs (logs/gaia/explanations/<config>/sample_*.json)
and checks whether each generated explanation only mentions services and error
types that are actually present in the evidence prompt it was given.

A *mention* is a service name or error type referenced in the explanation text
(evidence_summary + explanation). A mention is *unsupported* (a hallucination)
when it does not appear in the sample's `prompt` (the evidence the LLM saw).

GAIA has a closed vocabulary of 10 service instances and 3 trace error types,
so extraction is exact-match regex — no NLP heuristics needed.

Writes:
  logs/gaia/explanation_eval/<config>/faithfulness_per_sample.csv
  logs/gaia/explanation_eval/<config>/faithfulness_report.txt

Run (from TransTVDiag_attention_analysis/):
    python experimental/eval_faithfulness.py --config cb_ce_awl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

# Closed GAIA vocabulary.
SERVICE_FAMILIES = ("webservice", "mobservice", "dbservice", "redisservice", "logservice")
SERVICE_RE = re.compile(r"\b(" + "|".join(SERVICE_FAMILIES) + r")([12])?\b", re.IGNORECASE)

# Trace error types, as they appear in the prompt evidence.
ERROR_PATTERNS = {
    "PD": re.compile(r"performance degradation|\bPD\b|\bdegradation\b", re.IGNORECASE),
    "500": re.compile(r"\b500\b|server error", re.IGNORECASE),
    "400": re.compile(r"\b400\b|client error", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl")
    p.add_argument("--explanations_root", default="logs/gaia/explanations")
    p.add_argument("--out_root", default="logs/gaia/explanation_eval")
    return p.parse_args()


def extract_services(text: str) -> tuple[set[str], set[str]]:
    """Return (instance_mentions, family_mentions) found in text.

    'redisservice2' -> instance; 'redisservice' (no digit) -> family only.
    """
    instances, families = set(), set()
    for fam, digit in SERVICE_RE.findall(text):
        fam = fam.lower()
        families.add(fam)
        if digit:
            instances.add(f"{fam}{digit}")
    return instances, families


def extract_error_types(text: str) -> set[str]:
    return {name for name, pat in ERROR_PATTERNS.items() if pat.search(text)}


# Instance-level service name (digit required) — used for relational/modality checks.
INSTANCE_RE = re.compile(r"\b((?:web|mob|db|redis|log)service[12])\b", re.IGNORECASE)
MODALITY_RE = re.compile(r"\b(traces?|logs?|metrics?)\b", re.IGNORECASE)
NEG_RE = re.compile(
    r"\b(no|not|never|without|nor|none|lack|lacks|lacking|absence|absent)\b|n't",
    re.IGNORECASE)
FROM_TO_RE = re.compile(
    r"\b(from|to)\s+((?:web|mob|db|redis|log)service[12])\b", re.IGNORECASE)


def _modality(word: str) -> str:
    return word.lower().rstrip("s")  # traces->trace, logs->log, metrics->metric


# ---------- prompt-side structure (the ground truth for these checks) -------

def parse_prompt_trace_edges(prompt: str) -> set[tuple[str, str, str]]:
    """Directed error edges (src, dst, error) stated in the TRACE evidence.

    'outgoing PEER: ...' under focal F  -> (F, PEER, err)
    'incoming PEER: ...' under focal F  -> (PEER, F, err)
    """
    edges: set[tuple[str, str, str]] = set()
    section, focal = None, None
    for raw in prompt.splitlines():
        line = raw.strip()
        if line.startswith("--- TRACE"):
            section, focal = "trace", None
        elif line.startswith("--- LOG"):
            section, focal = "log", None
        elif line.startswith("--- METRIC"):
            section, focal = "metric", None
        elif line.startswith("*"):
            m = INSTANCE_RE.search(line)
            focal = m.group(1).lower() if m else None
        elif section == "trace" and focal and (
                line.startswith("incoming") or line.startswith("outgoing")):
            head, _, desc = line.partition(":")
            pm = INSTANCE_RE.search(head)
            if not pm:
                continue
            peer, direction = pm.group(1).lower(), head.split()[0].lower()
            for err in extract_error_types(desc):
                edges.add((peer, focal, err) if direction == "incoming"
                          else (focal, peer, err))
    return edges


def parse_prompt_modalities(prompt: str) -> dict[str, set[str]]:
    """service -> {modalities it has evidence in}.

    A service is "in trace" if it is a top-K trace service OR appears as a peer
    in a trace edge (peers are genuinely part of the trace evidence). For log and
    metric only the top-K header services count (their indented lines are raw text
    that may mention unrelated services).
    """
    svc2mod: dict[str, set[str]] = {}
    section = None
    for raw in prompt.splitlines():
        line = raw.strip()
        if line.startswith("--- TRACE"):
            section = "trace"
        elif line.startswith("--- LOG"):
            section = "log"
        elif line.startswith("--- METRIC"):
            section = "metric"
        elif line.startswith("*") and section:
            m = INSTANCE_RE.search(line)
            if m:
                svc2mod.setdefault(m.group(1).lower(), set()).add(section)
        elif section == "trace" and (
                line.startswith("incoming") or line.startswith("outgoing")):
            head = line.partition(":")[0]
            pm = INSTANCE_RE.search(head)
            if pm:
                svc2mod.setdefault(pm.group(1).lower(), set()).add("trace")
    return svc2mod


# ---------- explanation-side claim extraction (conservative) ---------------

def extract_relational_claims(text: str) -> list[tuple[str, str, str]]:
    """Directed (src, dst, error) edges asserted in the explanation text.

    Each 'from PEER'/'to PEER' is bound to the nearest service mentioned *before*
    it (the local subject), not the first service in the sentence. This avoids two
    systematic false positives seen on multi-hop sentences:
      * chain collapse: 'A sends X to B, which sends Y to C' must not yield A->C;
        the 'to C' phrase binds to B (its nearest preceding subject), giving B->C.
      * predicted-RC binding: '...mobservice1 sends a 400 to redisservice1' must
        bind to mobservice1, not to the predicted root cause named earlier.
    The error type is taken from the local window, or from the sentence when it
    has a single edge and a single error. Ambiguous claims are skipped so the
    check only fires on confidently parsed edges (precision over recall).
    """
    claims: list[tuple[str, str, str]] = []
    for sent in re.split(r"[.;]\s+", text):
        insts = [(m.start(), m.group(1).lower())
                 for m in INSTANCE_RE.finditer(sent)]
        if not insts:
            continue
        matches = list(FROM_TO_RE.finditer(sent))
        # Instances that are the object of a 'from/to PEER' phrase are not the
        # subject of the clause; the subject is everything else (the services the
        # sentence is talking *about*).
        obj_pos = {m.start(2) for m in matches}
        subjects = [(pos, name) for pos, name in insts if pos not in obj_pos]
        sent_errs = extract_error_types(sent)
        # Only borrow the sentence-level error when there is a single edge;
        # with several edges in one sentence the error must be bound locally.
        allow_sentence_err = len(matches) == 1
        prev_end, prev = 0, None  # prev = (direction, peer) of the last phrase
        for m in matches:
            window = sent[prev_end:m.start()]
            prev_end = m.end()
            direction, peer = m.group(1).lower(), m.group(2).lower()
            # Chain (A->B, which ...-> C) reads B->C: a 'to' right after another
            # 'to' binds to that previous destination. Otherwise the edge belongs
            # to the nearest preceding subject (skipping intervening from/to peers,
            # so 'X has errors from P and a 400 to C' gives X->C, not P->C).
            if direction == "to" and prev and prev[0] == "to":
                focal = prev[1]
            else:
                before = [name for pos, name in subjects if pos < m.start()]
                focal = before[-1] if before else None
            prev = (direction, peer)
            if not focal or focal == peer:
                continue
            werrs = extract_error_types(window)
            if len(werrs) == 1:
                err = next(iter(werrs))
            elif not werrs and allow_sentence_err and len(sent_errs) == 1:
                err = next(iter(sent_errs))
            else:
                continue  # cannot bind an error confidently -> skip
            claims.append((peer, focal, err) if direction == "from"
                          else (focal, peer, err))
    return claims


def extract_modality_claims(text: str) -> list[tuple[str, str]]:
    """(modality, service) attributions: 'log evidence ... redisservice1'.

    Skips negated contexts ('no trace evidence for X') so absence claims are
    not mistaken for misattribution.
    """
    claims: set[tuple[str, str]] = set()
    # Skip any sentence that carries a negation ('lacks log signals for X',
    # 'no trace evidence for X'): such absence claims must not be read as a
    # modality attribution, and the negation can sit a clause away from the
    # service. Within a clean sentence, bind each modality word to the nearest
    # service in the same clause (avoids cross-clause mispairing).
    for sent in re.split(r"[.;]\s+", text):
        if NEG_RE.search(sent):
            continue
        for clause in re.split(r"[,]|\band\b", sent, flags=re.IGNORECASE):
            mods = [(m.start(), _modality(m.group(1)))
                    for m in MODALITY_RE.finditer(clause)]
            svcs = [(s.start(), s.group(1).lower())
                    for s in INSTANCE_RE.finditer(clause)]
            if not mods or not svcs:
                continue
            for mpos, mod in mods:
                _, svc = min(svcs, key=lambda ps: abs(ps[0] - mpos))
                claims.add((mod, svc))
    return list(claims)


def evaluate_sample(sample: dict) -> dict | None:
    parsed = sample.get("parsed")
    if not parsed:
        return None  # generation failed (status != ok / unparseable)
    prompt = sample.get("prompt", "")
    expl_text = " ".join([
        parsed.get("evidence_summary", "") or "",
        parsed.get("explanation", "") or "",
    ])

    prompt_inst, prompt_fam = extract_services(prompt)
    prompt_err = extract_error_types(prompt)
    expl_inst, expl_fam = extract_services(expl_text)
    expl_err = extract_error_types(expl_text)

    # Service mentions: instance unsupported if not in prompt instances; a bare
    # family mention is supported if any instance of that family is in the prompt.
    bad_inst = {s for s in expl_inst if s not in prompt_inst}
    bad_fam = {f for f in expl_fam if f not in prompt_fam}
    bad_err = {e for e in expl_err if e not in prompt_err}

    total_mentions = len(expl_inst) + len(expl_fam) + len(expl_err)
    bad_mentions = len(bad_inst) + len(bad_fam) + len(bad_err)

    # Relational faithfulness: directed (src, dst, error) edges.
    prompt_edges = parse_prompt_trace_edges(prompt)
    rel_claims = extract_relational_claims(expl_text)
    rel_bad = [c for c in rel_claims if c not in prompt_edges]

    # Modality-attribution faithfulness: (modality, service) bindings.
    prompt_mods = parse_prompt_modalities(prompt)
    mod_claims = extract_modality_claims(expl_text)
    mod_bad = [c for c in mod_claims if c[0] not in prompt_mods.get(c[1], set())]

    pred = sample.get("rcl_pred_service", "")
    return {
        "gaia_index": sample.get("gaia_index"),
        "group": sample.get("group"),
        "rcl_correct": sample.get("rcl_correct"),
        "rcl_agreement": parsed.get("rcl_agreement"),
        "total_mentions": total_mentions,
        "bad_mentions": bad_mentions,
        "fully_grounded": bad_mentions == 0,
        "mentions_pred_rcl": pred in expl_inst,
        "hallucinated_services": ",".join(sorted(bad_inst | bad_fam)),
        "hallucinated_errors": ",".join(sorted(bad_err)),
        "relational_claims": len(rel_claims),
        "relational_bad": len(rel_bad),
        "modality_claims": len(mod_claims),
        "modality_bad": len(mod_bad),
        "bad_edges": " | ".join(f"{s}->{d}:{e}" for s, d, e in rel_bad),
        "bad_modality": " | ".join(f"{m}:{s}" for m, s in mod_bad),
        "fully_grounded_strict": (bad_mentions == 0 and not rel_bad and not mod_bad),
    }


def main() -> None:
    args = parse_args()
    expl_dir = Path(args.explanations_root) / args.config
    out_dir = Path(args.out_root) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = sorted(expl_dir.glob("sample_*.json"))
    if not sample_paths:
        raise SystemExit(f"no sample_*.json found in {expl_dir}")

    rows, skipped, seen = [], 0, set()
    for path in sample_paths:
        with open(path) as f:
            sample = json.load(f)
        gi = sample.get("gaia_index")
        if gi in seen:
            continue  # ignore duplicate copies (e.g. "sample_X (1).json")
        seen.add(gi)
        result = evaluate_sample(sample)
        if result is None:
            skipped += 1
            continue
        rows.append(result)

    if not rows:
        raise SystemExit("no parseable samples to evaluate")

    per_sample_path = out_dir / "faithfulness_per_sample.csv"
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Corpus aggregates.
    n = len(rows)
    total_mentions = sum(r["total_mentions"] for r in rows)
    bad_mentions = sum(r["bad_mentions"] for r in rows)
    hallucination_rate = bad_mentions / total_mentions if total_mentions else 0.0
    fully_grounded_rate = sum(r["fully_grounded"] for r in rows) / n
    mentions_pred_rate = sum(r["mentions_pred_rcl"] for r in rows) / n

    rel_claims = sum(r["relational_claims"] for r in rows)
    rel_bad = sum(r["relational_bad"] for r in rows)
    rel_rate = rel_bad / rel_claims if rel_claims else 0.0
    mod_claims = sum(r["modality_claims"] for r in rows)
    mod_bad = sum(r["modality_bad"] for r in rows)
    mod_rate = mod_bad / mod_claims if mod_claims else 0.0
    strict_rate = sum(r["fully_grounded_strict"] for r in rows) / n

    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)

    lines = []
    lines.append("=== Faithfulness / hallucination report ===")
    lines.append(f"config: {args.config}")
    lines.append(f"samples evaluated: {n}  (skipped/unparseable: {skipped})")
    lines.append("")
    lines.append(f"total mentions:        {total_mentions}")
    lines.append(f"unsupported mentions:  {bad_mentions}")
    lines.append(f"hallucination_rate:    {hallucination_rate:.4f}  "
                 f"(unsupported / total mentions)")
    lines.append(f"fully_grounded_rate:   {fully_grounded_rate:.4f}  "
                 f"(explanations with 0 hallucinations)")
    lines.append(f"mentions_pred_rcl:     {mentions_pred_rate:.4f}  "
                 f"(explanation names the predicted root cause)")
    lines.append("")
    lines.append("-- atom-level above (closed vocab); relational/modality below --")
    lines.append(f"relational claims:     {rel_claims}  (directed src->dst:error edges)")
    lines.append(f"relational_bad:        {rel_bad}")
    lines.append(f"relational_hallucination_rate: {rel_rate:.4f}  "
                 f"(edge stated but not in evidence: wrong peer/direction/binding)")
    lines.append(f"modality claims:       {mod_claims}  (modality->service attributions)")
    lines.append(f"modality_bad:          {mod_bad}")
    lines.append(f"modality_misattribution_rate:  {mod_rate:.4f}  "
                 f"(service credited to a modality it has no evidence in)")
    lines.append(f"fully_grounded_strict: {strict_rate:.4f}  "
                 f"(0 atom + 0 relational + 0 modality errors)")
    lines.append("")
    lines.append("by group:")
    lines.append(f"  {'group':<26} {'n':>4} {'halluc_rate':>12} {'grounded':>10}")
    for g in sorted(by_group):
        grp = by_group[g]
        tm = sum(r["total_mentions"] for r in grp)
        bm = sum(r["bad_mentions"] for r in grp)
        hr = bm / tm if tm else 0.0
        gr = sum(r["fully_grounded"] for r in grp) / len(grp)
        lines.append(f"  {g:<26} {len(grp):>4} {hr:>12.4f} {gr:>10.4f}")
    lines.append("")
    worst = Counter()
    for r in rows:
        for s in filter(None, r["hallucinated_services"].split(",")):
            worst[s] += 1
    if worst:
        lines.append("most hallucinated services: "
                     + ", ".join(f"{s}({c})" for s, c in worst.most_common(10)))
    else:
        lines.append("no hallucinated services detected.")

    report = "\n".join(lines)
    (out_dir / "faithfulness_report.txt").write_text(report + "\n")
    print(report)
    print(f"\nwrote {per_sample_path}")


if __name__ == "__main__":
    main()
