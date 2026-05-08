"""Evidence ranking from attention + event JSONs.

For one sample we know:
  - cls_attns[modality] = attention vector of shape (N_real,) — what the
    [CLS]-aggregator looked at, head-mean of attn[layer=0, 0, 1:]
  - rcl_pred, fti_pred — model's predictions
  - events/{log,trace,metric}.json — extracted events for this sample
  - log_per_sample_lookup.json (optional) — raw log examples for ~6% of
    samples; falls back to drain template text otherwise

This module produces a structured evidence bundle suitable as input to
an SRE-facing explanation step (LLM prompt or thesis case-study figure).

Top-K ranking is over the flattened (3 modalities x N_real services) grid
of CLS attention, picking the strongest pairs. For each pair we fetch
matching raw events from the right event file and aggregate them into a
short summary.

Two entry points:

  build_evidence_bundle(sample_idx, cls_attns, ...) — pure function.
      Used both by the offline batch-aggregator (cls_attns from h5) and
      by the online diagnose() function (cls_attns from forward pass).

  build_evidence_bundle_from_h5(h5_path, csv_path, sample_idx_in_fold)
      — convenience wrapper for offline use against a single h5.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


MODALITIES = ("metric", "trace", "log")
ERROR_NAMES = {"PD": "performance degradation",
               "500": "HTTP 500 server error",
               "400": "HTTP 400 client error"}


# ---------- Loaders -----------------------------------------------------

def load_event_files(events_dir: Path) -> dict:
    """Load log/metric/trace event JSONs once and reuse across samples."""
    events_dir = Path(events_dir)
    out = {}
    for m in MODALITIES:
        with open(events_dir / f"{m}.json") as f:
            out[m] = json.load(f)
    return out


def load_log_lookup(path: Optional[Path]) -> Optional[dict]:
    """Load optional raw-log lookup. Returns None if path is None or missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------- Per-modality evidence collectors ----------------------------

def collect_log_evidence(
    gaia_index: int,
    service: str,
    log_events: Sequence,
    log_lookup: Optional[dict],
    max_templates: int = 3,
    max_raw_lines: int = 2,
) -> dict:
    """Aggregate log events for one (sample, service).

    log_events is the per-sample list ``[[svc, tid], ...]``. We filter to
    rows where svc matches, count by template_id, take the top
    max_templates, and for each one attach template text plus (if
    available) up to max_raw_lines raw examples from log_lookup.

    ``gaia_index`` is the global GAIA row id (0..16204) — log_lookup is
    keyed on it, NOT on the fold-local sample_idx.
    """
    matching_tids = [str(tid) for svc, tid in log_events if svc == service]
    if not matching_tids:
        return {"n_events": 0, "templates": []}

    counts = Counter(matching_tids)
    templates_out = []
    sample_bucket = (log_lookup["samples"].get(str(gaia_index), {})
                     if log_lookup else {})
    template_text_map = (log_lookup["templates"] if log_lookup else {})

    for tid, count in counts.most_common(max_templates):
        entry = {
            "template_id": int(tid),
            "count": count,
            "template_text": template_text_map.get(tid),
        }
        key = f"{service}__{tid}"
        if key in sample_bucket:
            entry["raw_examples"] = sample_bucket[key][:max_raw_lines]
        templates_out.append(entry)

    return {"n_events": len(matching_tids), "templates": templates_out}


def collect_trace_evidence(
    service: str,
    trace_events: Sequence,
    max_edges: int = 3,
) -> dict:
    """Aggregate trace events touching this service (as callee or caller)."""
    matching = []
    for event in trace_events:
        callee, caller, err = event[0], event[1], event[2]
        if service == callee:
            matching.append(("incoming", caller, err))
        elif service == caller:
            matching.append(("outgoing", callee, err))
    if not matching:
        return {"n_events": 0, "edges": []}

    counts = Counter(matching)
    edges_out = []
    for (direction, peer, err), count in counts.most_common(max_edges):
        edges_out.append({
            "direction": direction,
            "peer_service": peer,
            "error_type": err,
            "error_description": ERROR_NAMES.get(err, err),
            "count": count,
        })
    return {"n_events": len(matching), "edges": edges_out}


def collect_metric_evidence(
    service: str,
    metric_events: Sequence,
    max_metrics: int = 5,
) -> dict:
    """Aggregate metric anomalies for this service (any host)."""
    matching = [(host, name) for svc, host, name in metric_events
                if svc == service]
    if not matching:
        return {"n_events": 0, "anomalies": []}

    by_metric = Counter(name for _, name in matching)
    hosts = sorted({host for host, _ in matching})
    anomalies_out = [
        {"metric_name": name, "count": count}
        for name, count in by_metric.most_common(max_metrics)
    ]
    return {"n_events": len(matching), "hosts": hosts,
            "anomalies": anomalies_out}


# ---------- Top-K ranking + bundle assembly -----------------------------

def _flatten_attention(cls_attns: Dict[str, np.ndarray]) -> List[tuple]:
    """Return list of (modality, service_idx, attention_weight) sorted desc."""
    rows = []
    for m in MODALITIES:
        if m not in cls_attns:
            continue
        vec = np.asarray(cls_attns[m], dtype=np.float32)
        for svc_idx, val in enumerate(vec):
            rows.append((m, svc_idx, float(val)))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def build_evidence_bundle(
    sample_idx: int,
    gaia_index: int,
    cls_attns: Dict[str, np.ndarray],
    rcl_pred: int,
    fti_pred: str,
    instance_names: Sequence[str],
    events: dict,
    log_lookup: Optional[dict] = None,
    top_k: int = 5,
) -> dict:
    """Build a structured evidence package for one sample.

    Parameters
    ----------
    sample_idx : int
        Fold-local position (0..3240). Used only as a label in the
        output bundle — does NOT key any events file.
    gaia_index : int
        Global GAIA row id (0..16204). All events JSONs and the raw-log
        lookup are keyed on this. The same gaia_index identifies the
        sample across folds.
    cls_attns : dict
        ``{modality: ndarray of shape (N_real,)}`` — CLS-row, head-mean,
        layer 0. Sums to <=1 per row.
    rcl_pred : int
        Predicted root-cause node index.
    fti_pred : str
        Predicted failure-type label.
    instance_names : sequence of str
        Maps node index -> service name.
    events : dict
        ``{"log": ..., "metric": ..., "trace": ...}`` from
        :func:`load_event_files`. Each is a per-sample dict.
    log_lookup : dict or None
        Output of :func:`load_log_lookup`. Optional; if provided and the
        sample is covered, evidence will include raw log lines.
    top_k : int
        Number of (modality, service) pairs to report.
    """
    gid_str = str(gaia_index)
    log_events = events["log"].get(gid_str, [])
    trace_events = events["trace"].get(gid_str, [])
    metric_events = events["metric"].get(gid_str, [])

    n_real = len(instance_names)
    uniform_baseline = 1.0 / n_real

    flat = _flatten_attention(cls_attns)
    evidence_list = []
    for rank, (modality, svc_idx, weight) in enumerate(flat[:top_k], start=1):
        service = instance_names[svc_idx]
        entry = {
            "rank": rank,
            "modality": modality,
            "service": service,
            "service_idx": svc_idx,
            "attention_weight": weight,
            "attention_lift": weight / uniform_baseline,
            "is_predicted_rcl": svc_idx == rcl_pred,
        }
        if modality == "log":
            entry["details"] = collect_log_evidence(
                gaia_index, service, log_events, log_lookup)
        elif modality == "trace":
            entry["details"] = collect_trace_evidence(service, trace_events)
        elif modality == "metric":
            entry["details"] = collect_metric_evidence(service, metric_events)
        evidence_list.append(entry)

    return {
        "sample_idx": int(sample_idx),
        "gaia_index": int(gaia_index),
        "fti_pred": fti_pred,
        "rcl_pred": int(rcl_pred),
        "rcl_pred_service": instance_names[rcl_pred],
        "n_real_services": n_real,
        "uniform_baseline": uniform_baseline,
        "top_k": top_k,
        "evidence": evidence_list,
    }


# ---------- Per-modality bundle (always emits log/trace/metric) ---------

def build_evidence_bundle_per_modality(
    sample_idx: int,
    gaia_index: int,
    cls_attns: Dict[str, np.ndarray],
    rcl_pred: int,
    fti_pred: str,
    instance_names: Sequence[str],
    events: dict,
    log_lookup: Optional[dict] = None,
    top_k_per_modality: int = 3,
) -> dict:
    """Build a per-modality evidence package for one sample.

    Same inputs as :func:`build_evidence_bundle`, but the output always
    contains one section per modality (log/trace/metric). Each section
    holds the top-K services *for that modality* by CLS attention weight,
    with full ``details`` populated. Empty event lists are kept rather
    than dropped, so every bundle has the same shape — convenient for
    downstream batch consumers (LLM prompt, aggregate stats).

    ``sample_idx`` is the fold-local position (label only); ``gaia_index``
    is the global row id used to look up events and raw logs.
    """
    gid_str = str(gaia_index)
    per_modality_events = {
        "log": events["log"].get(gid_str, []),
        "trace": events["trace"].get(gid_str, []),
        "metric": events["metric"].get(gid_str, []),
    }

    n_real = len(instance_names)
    uniform_baseline = 1.0 / n_real

    evidence_by_modality: Dict[str, list] = {}
    for modality in MODALITIES:
        entries: list = []
        if modality in cls_attns:
            vec = np.asarray(cls_attns[modality], dtype=np.float32)
            order = np.argsort(-vec)[:top_k_per_modality]
            for rank, svc_idx in enumerate(order, start=1):
                svc_idx = int(svc_idx)
                service = instance_names[svc_idx]
                weight = float(vec[svc_idx])
                entry = {
                    "rank": rank,
                    "service": service,
                    "service_idx": svc_idx,
                    "attention_weight": weight,
                    "attention_lift": weight / uniform_baseline,
                    "is_predicted_rcl": svc_idx == rcl_pred,
                }
                if modality == "log":
                    entry["details"] = collect_log_evidence(
                        gaia_index, service,
                        per_modality_events["log"], log_lookup)
                elif modality == "trace":
                    entry["details"] = collect_trace_evidence(
                        service, per_modality_events["trace"])
                elif modality == "metric":
                    entry["details"] = collect_metric_evidence(
                        service, per_modality_events["metric"])
                entries.append(entry)
        evidence_by_modality[modality] = entries

    return {
        "sample_idx": int(sample_idx),
        "gaia_index": int(gaia_index),
        "fti_pred": fti_pred,
        "rcl_pred": int(rcl_pred),
        "rcl_pred_service": instance_names[rcl_pred],
        "n_real_services": n_real,
        "uniform_baseline": uniform_baseline,
        "top_k_per_modality": top_k_per_modality,
        "evidence_by_modality": evidence_by_modality,
    }


# ---------- h5-driven convenience wrapper -------------------------------

def build_evidence_bundle_from_h5(
    h5_path: Path,
    index_csv_path: Path,
    sample_idx_in_fold: int,
    events: dict,
    log_lookup: Optional[dict] = None,
    top_k: int = 5,
    layer: int = 0,
) -> dict:
    """Convenience wrapper: read attention + predictions for one sample
    from a fold's h5 + sidecar CSV, then call :func:`build_evidence_bundle`.

    Use for offline case studies on already-extracted folds. For online
    diagnosis, prefer :func:`build_evidence_bundle` directly with
    captured attention from the live forward pass.
    """
    import h5py
    import pandas as pd

    with h5py.File(h5_path, "r") as h5:
        instance_names = [
            (n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
            for n in h5.attrs.get("instance_names", [f"node_{i}" for i in range(10)])
        ]
        cls_attns = {}
        for m in MODALITIES:
            attn = np.asarray(h5[m]["attn_post"][layer, sample_idx_in_fold],
                              dtype=np.float32)
            cls_attns[m] = attn.mean(axis=0)[0, 1:]

    df = pd.read_csv(index_csv_path)
    row = df.iloc[sample_idx_in_fold].to_dict()
    return build_evidence_bundle(
        sample_idx=int(row["sample_idx"]),
        gaia_index=int(row["gaia_index"]),
        cls_attns=cls_attns,
        rcl_pred=int(row["rcl_pred"]),
        fti_pred=str(row.get("fti_pred", "")),
        instance_names=instance_names,
        events=events,
        log_lookup=log_lookup,
        top_k=top_k,
    )
