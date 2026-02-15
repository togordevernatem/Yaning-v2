#!/usr/bin/env python3
"""tools/export_alignment_from_ours_metrics.py

Export an "alignment JSON" from Ours' saved metrics.json.

Why
---
Your repo has two result formats:
  - GRU baseline: logs/*.npz (now includes split/meta fields)
  - Ours: logs/table_runs/**/metrics.json

To compare apples-to-apples, we need a shared, explicit contract describing:
  - mode
  - split ratios
  - total/train/val/test sizes (where possible)
  - test next-event count
  - seen/ood next-event counts

This script reads an Ours metrics.json and emits a JSON following the same core
fields as tools/check_protocol_alignment.py, so it can be compared against:
  - the canonical alignment report (from dataset)
  - GRU baseline npz meta

Important notes
--------------
Ours metrics.json stores counts for test next-event positions:
  summary.test.all.count  == n_test_next_event
  summary.test.seen.count == seen_next_event_count
  summary.test.ood.count  == ood_next_event_count

It does NOT store n_total/n_train/n_val/n_test lengths directly. This script
therefore:
  - ALWAYS exports next-event counts from metrics.json
  - Optionally computes full sizes from the dataset if you pass --compute_full_sizes

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import torch

# Ensure repo root on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.check_protocol_alignment import compute_alignment


def _load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export alignment JSON from Ours metrics.json")
    ap.add_argument("--metrics", required=True, help="Path to Ours metrics.json")
    ap.add_argument("--out", default=None, help="Output JSON path. Default: logs/alignment_from_ours_{model}_seed{seed}.json")

    # These should match Ours code defaults; only used when computing full sizes.
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--snapshots_dir", default="data/snapshots")
    ap.add_argument("--events_dir", default="data/events")
    ap.add_argument("--truncate_icews_real_to", type=int, default=5000)

    ap.add_argument(
        "--compute_full_sizes",
        action="store_true",
        help="If set, compute n_total/n_train/n_val/n_test from GC_TPP_Dataset using the given split ratios.",
    )
    args = ap.parse_args()

    m = _load_metrics(args.metrics)
    meta = m.get("meta", {})
    summ = (((m.get("summary", {}) or {}).get("test", {}) or {}))

    mode = meta.get("data_mode")
    model = meta.get("model")
    seed = meta.get("seed")
    protocol = meta.get("protocol")

    if mode is None:
        raise KeyError("metrics.json missing meta.data_mode")

    # next-event counts as saved by Ours
    n_test_next_event = int(summ.get("all", {}).get("count"))
    seen_next_event_count = int(summ.get("seen", {}).get("count"))
    ood_next_event_count = int(summ.get("ood", {}).get("count"))

    out = {
        "mode": mode,
        "protocol": protocol,
        "model": model,
        "seed": seed,
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "n_test_next_event": n_test_next_event,
        "seen_next_event_count": seen_next_event_count,
        "ood_next_event_count": ood_next_event_count,
        "seen_ood_key": "(src, dst, coarse_type)",
        "flags_semantics": "ours metrics.json counts are next-event positions",
        "source_metrics": args.metrics,
    }

    if args.compute_full_sizes:
        rep = compute_alignment(
            mode=mode,
            snapshots_dir=args.snapshots_dir,
            events_dir=args.events_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            truncate_icews_real_to=args.truncate_icews_real_to,
            device=torch.device("cpu"),
        )
        out.update(
            {
                "snapshots_dir": rep.snapshots_dir,
                "events_dir": rep.events_dir,
                "T": rep.T,
                "N": rep.N,
                "F_in": rep.F_in,
                "n_total": rep.n_total,
                "n_train": rep.n_train,
                "n_val": rep.n_val,
                "n_test": rep.n_test,
                # for cross-check
                "alignment_from_dataset": os.path.basename(args.metrics) + ":computed",
            }
        )

    if args.out is None:
        os.makedirs("logs", exist_ok=True)
        safe_model = (model or "unknown_model")
        safe_seed = (seed if seed is not None else "NA")
        out_path = os.path.join("logs", f"alignment_from_ours_{safe_model}_seed{safe_seed}.json")
    else:
        out_path = args.out

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out_path)
    print("      mode=", mode, "protocol=", protocol, "model=", model, "seed=", seed)
    print("      next-event counts: total=", n_test_next_event, "seen=", seen_next_event_count, "ood=", ood_next_event_count)


if __name__ == "__main__":
    main()

