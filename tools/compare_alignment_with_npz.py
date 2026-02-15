#!/usr/bin/env python3
"""tools/compare_alignment_with_npz.py

Compare a protocol alignment JSON report against a GRU baseline npz.

Why
---
Even with best intentions, it's easy to accidentally compare results under
slightly different split ratios / dataset modes / seen-ood semantics.

This tool makes it crystal clear:
  - MATCH: all key alignment fields exactly match
  - MISMATCH: prints field-level diffs (expected vs got)

Inputs
------
- alignment JSON: produced by tools/check_protocol_alignment.py
- baseline npz: produced by main_gru_lognormal_baseline.py

Example
-------
python tools/compare_alignment_with_npz.py \
  --alignment logs/alignment_icews_real_topk500_K500_tr0p7_va0p15.json \
  --npz logs/gru_lognormal_icews_real_topk500_K500_seed0.npz

Exit codes
----------
0: MATCH
2: MISMATCH

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _npz_value(d: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in npz")
    v = d[key]
    # Scalars saved by np.savez may come back as 0-d arrays
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def _to_primitive(x: Any) -> Any:
    """Convert numpy scalar/array to python primitive for comparison/printing."""
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        # for arrays keep shape+dtype summary
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (np.generic,)):
        return x.item()
    return x


def _compare_fields(aln: Dict[str, Any], npz: np.lib.npyio.NpzFile) -> List[Tuple[str, Any, Any]]:
    """Return list of diffs: (field, expected, got)."""

    # These fields are the canonical contract for alignment.
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

    for j_key, z_key in fields:
        if j_key not in aln:
            diffs.append((j_key, "<missing in alignment json>", _to_primitive(_npz_value(npz, z_key))))
            continue

        exp = aln[j_key]
        got = _npz_value(npz, z_key)

        # Normalize numeric types
        if isinstance(exp, (int, float)) and isinstance(got, (int, float, np.generic)):
            got_v = float(got) if isinstance(exp, float) else int(got)
            if got_v != exp:
                diffs.append((j_key, exp, got_v))
            continue

        got_p = _to_primitive(got)
        if exp != got_p:
            diffs.append((j_key, exp, got_p))

    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare alignment JSON vs GRU baseline npz")
    ap.add_argument("--alignment", required=True, help="Path to logs/alignment_*.json")
    ap.add_argument("--npz", required=True, help="Path to logs/gru_lognormal_*.npz")
    ap.add_argument("--strict", action="store_true", help="If set, also require alignment JSON seen_ood_key == '(src, dst, coarse_type)'.")
    args = ap.parse_args()

    if not os.path.exists(args.alignment):
        raise FileNotFoundError(args.alignment)
    if not os.path.exists(args.npz):
        raise FileNotFoundError(args.npz)

    aln = _load_json(args.alignment)

    with np.load(args.npz, allow_pickle=True) as z:
        diffs = _compare_fields(aln, z)

        if args.strict:
            key = aln.get("seen_ood_key", None)
            if key is None:
                diffs.append(("seen_ood_key", "(src, dst, coarse_type)", "<missing in alignment json>"))
            elif key != "(src, dst, coarse_type)":
                diffs.append(("seen_ood_key", "(src, dst, coarse_type)", key))

    # Report
    print("=" * 72)
    print("[COMPARE ALIGNMENT vs NPZ]")
    print("alignment =", args.alignment)
    print("npz       =", args.npz)

    if not diffs:
        print("RESULT: MATCH")
        print("All checked fields match exactly.")
        print("=" * 72)
        raise SystemExit(0)

    print("RESULT: MISMATCH")
    print("Diffs:")
    for field, exp, got in diffs:
        print(f"- {field}: expected={exp} got={got}")
    print("=" * 72)
    raise SystemExit(2)


if __name__ == "__main__":
    main()

