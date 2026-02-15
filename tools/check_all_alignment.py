#!/usr/bin/env python3
"""tools/check_all_alignment.py

One-shot end-to-end alignment gate.

What it does
------------
Given:
  - dataset mode + split ratios
  - a GRU baseline npz (logs/gru_lognormal_*.npz)
  - an Ours metrics.json (logs/**/metrics.json)

It will:
  1) generate (or reuse) canonical alignment JSON from dataset
  2) export alignment JSON from Ours metrics.json
  3) compare canonical alignment vs GRU npz
  4) compare canonical alignment vs Ours-exported alignment

Exit codes
----------
0: ALL MATCH
2: ANY MISMATCH

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure repo root on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.check_protocol_alignment import compute_alignment


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _npz_get(npz: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in npz:
        raise KeyError(f"missing key in npz: {key}")
    v = npz[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def _compare_alignment_json_to_npz(aln: Dict[str, Any], npz_path: str) -> List[Tuple[str, Any, Any]]:
    """Compare canonical alignment JSON -> npz (GRU baseline).

    Returns diffs as list of (field, expected, got).
    """
    fields = [
        ("mode", "mode"),
        ("train_ratio", "train_ratio"),
        ("val_ratio", "val_ratio"),
        ("n_total", "n_total"),
        ("n_train", "n_train"),
        ("n_val", "n_val"),
        ("n_test", "n_test"),
        ("n_test_next_event", "n_test_next_event"),
        ("seen_next_event_count", "seen_next_event_count"),
        ("ood_next_event_count", "ood_next_event_count"),
    ]

    diffs: List[Tuple[str, Any, Any]] = []
    with np.load(npz_path, allow_pickle=True) as z:
        for j_key, z_key in fields:
            exp = aln.get(j_key, None)
            if exp is None:
                diffs.append((j_key, "<missing in alignment json>", _npz_get(z, z_key)))
                continue
            got = _npz_get(z, z_key)
            # normalize scalar types
            if isinstance(got, np.ndarray):
                if got.shape == ():
                    got = got.item()
            if isinstance(exp, float):
                got_v = float(got)
            elif isinstance(exp, int):
                got_v = int(got)
            else:
                got_v = got
            if got_v != exp:
                diffs.append((j_key, exp, got_v))

    return diffs


def _compare_alignment_jsons(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """Compare canonical alignment JSON -> exported ours-alignment JSON.

    Returns diffs as list of (field, expected, got).
    """
    fields = [
        "mode",
        "train_ratio",
        "val_ratio",
        "n_total",
        "n_train",
        "n_val",
        "n_test",
        "n_test_next_event",
        "seen_next_event_count",
        "ood_next_event_count",
    ]
    diffs: List[Tuple[str, Any, Any]] = []
    for k in fields:
        exp = a.get(k, None)
        got = b.get(k, None)
        if exp != got:
            diffs.append((k, exp, got))
    return diffs


def export_alignment_from_ours_metrics(
    metrics_path: str,
    train_ratio: float,
    val_ratio: float,
    canonical: Dict[str, Any],
) -> Dict[str, Any]:
    """Create an alignment dict from Ours metrics.json.

    Uses canonical to fill full sizes (n_total/n_train/n_val/n_test) to make
    comparisons strict.
    """
    m = _load_json(metrics_path)
    meta = m.get("meta", {})
    summ = (((m.get("summary", {}) or {}).get("test", {}) or {}))

    mode = meta.get("data_mode")
    if mode is None:
        raise KeyError("metrics.json missing meta.data_mode")

    out = {
        "mode": mode,
        "protocol": meta.get("protocol"),
        "model": meta.get("model"),
        "seed": meta.get("seed"),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "n_test_next_event": int(summ.get("all", {}).get("count")),
        "seen_next_event_count": int(summ.get("seen", {}).get("count")),
        "ood_next_event_count": int(summ.get("ood", {}).get("count")),
        "seen_ood_key": "(src, dst, coarse_type)",
        "flags_semantics": "ours metrics.json counts are next-event positions",
        "source_metrics": metrics_path,
    }

    # Fill in full sizes from canonical alignment (dataset)
    for k in ["n_total", "n_train", "n_val", "n_test", "T", "N", "F_in", "snapshots_dir", "events_dir"]:
        if k in canonical:
            out[k] = canonical[k]

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end alignment gate (dataset vs GRU vs Ours)")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--truncate_icews_real_to", type=int, default=5000)

    ap.add_argument("--gru_npz", required=True, help="Path to logs/gru_lognormal_*.npz")
    ap.add_argument("--ours_metrics", required=True, help="Path to Ours metrics.json")

    ap.add_argument("--snapshots_dir", default="data/snapshots")
    ap.add_argument("--events_dir", default="data/events")

    ap.add_argument("--out_dir", default="logs", help="Where to write generated alignment jsons")
    ap.add_argument("--no_write", action="store_true", help="Do not write json artifacts")
    args = ap.parse_args()

    # 1) canonical alignment (from dataset)
    rep = compute_alignment(
        mode=args.mode,
        snapshots_dir=args.snapshots_dir,
        events_dir=args.events_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        truncate_icews_real_to=args.truncate_icews_real_to,
        device=torch.device("cpu"),
    )

    canonical = rep.__dict__.copy()

    # 2) ours alignment + meta
    ours_metrics_obj = _load_json(args.ours_metrics)
    ours_meta = ours_metrics_obj.get("meta", {})

    ours_aln = export_alignment_from_ours_metrics(
        metrics_path=args.ours_metrics,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        canonical=canonical,
    )

    # derive output paths
    tr = str(args.train_ratio).replace(".", "p")
    va = str(args.val_ratio).replace(".", "p")
    canonical_path = os.path.join(args.out_dir, f"alignment_{args.mode}_tr{tr}_va{va}.json")

    ours_model = ours_meta.get("model", "unknown_model")
    ours_seed = ours_meta.get("seed", "NA")
    ours_path = os.path.join(args.out_dir, f"alignment_from_ours_{ours_model}_seed{ours_seed}.json")

    if not args.no_write:
        _write_json(canonical_path, canonical)
        _write_json(ours_path, ours_aln)

    # 3) compare canonical vs GRU
    diffs_gru = _compare_alignment_json_to_npz(canonical, args.gru_npz)

    # 4) compare canonical vs ours
    diffs_ours = _compare_alignment_jsons(canonical, ours_aln)

    ok = (len(diffs_gru) == 0) and (len(diffs_ours) == 0)

    print("=" * 72)
    print("[CHECK ALL ALIGNMENT]")
    print(f"mode={args.mode} train_ratio={args.train_ratio} val_ratio={args.val_ratio}")
    print("canonical:", canonical_path if not args.no_write else "<not written>")
    print("gru_npz  :", args.gru_npz)
    print("ours_met :", args.ours_metrics)
    print("ours_aln :", ours_path if not args.no_write else "<not written>")
    print("-")

    if not diffs_gru:
        print("GRU  : MATCH")
    else:
        print("GRU  : MISMATCH")
        for k, exp, got in diffs_gru:
            print(f"  - {k}: expected={exp} got={got}")

    if not diffs_ours:
        print("OURS : MATCH")
    else:
        print("OURS : MISMATCH")
        for k, exp, got in diffs_ours:
            print(f"  - {k}: expected={exp} got={got}")

    print("-")
    print("RESULT:", "ALL MATCH" if ok else "MISMATCH")
    print("=" * 72)

    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

