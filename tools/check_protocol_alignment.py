#!/usr/bin/env python3
"""tools/check_protocol_alignment.py

One-shot protocol alignment checker.

Goal
----
Make it impossible to accidentally compare baselines/models under different:
  - dataset mode
  - split ratios (train/val/test)
  - Seen/OOD definition (triplet key)
  - metric space (logΔt)

This script inspects the repository's canonical dataset implementation:
  data/dataset_toy.py::GC_TPP_Dataset
and reports:
  - total/train/val/test sizes
  - next-event count (len(test)-1)
  - seen/ood counts on next-event positions (aligned to dataset flags semantics)

Notes
-----
In this repo, GC_TPP_Dataset.get_seen_ood_flags(...) returns PER-SPLIT boolean flags:
  flags['seen_test'].shape == (len(idx_test),)
So next-event at step i corresponds to flags['seen_test'][i+1].

Example
-------
python tools/check_protocol_alignment.py --mode icews_real_topk500_K500 --train_ratio 0.7 --val_ratio 0.15

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

import torch

# Ensure repo root is on sys.path so `import data.*` works when running from tools/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.dataset_toy import GC_TPP_Dataset


@dataclass
class AlignmentReport:
    mode: str
    snapshots_dir: str
    events_dir: str
    T: int
    N: int
    F_in: int
    train_ratio: float
    val_ratio: float
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    n_test_next_event: int
    seen_test_count: int
    ood_test_count: int
    seen_next_event_count: int
    ood_next_event_count: int
    seen_ood_key: str
    flags_semantics: str


def _infer_snapshot_config(snapshots_dir: str) -> tuple[int, int, int]:
    """Infer T, N, F_in from snapshots_dir.

    Supports the repo's common formats:
      - data/snapshots/x_*.pt where x_0.pt is a dict {'x': Tensor[N,F], 'edge_index': ...}
      - or x_0.pt is directly a Tensor[N,F]

    Falls back to (T=20, N=10, F_in=3) if directory is missing.
    """
    if not os.path.isdir(snapshots_dir):
        return 20, 10, 3

    x_files = sorted([f for f in os.listdir(snapshots_dir) if f.startswith("x_") and f.endswith(".pt")])
    if not x_files:
        return 20, 10, 3

    T = len(x_files)
    x0_path = os.path.join(snapshots_dir, "x_0.pt")
    if not os.path.exists(x0_path):
        return T, 10, 3

    obj = torch.load(x0_path, map_location="cpu")
    if isinstance(obj, dict) and "x" in obj:
        x0 = obj["x"]
    else:
        x0 = obj

    if not isinstance(x0, torch.Tensor) or x0.dim() != 2:
        return T, 10, 3

    N = int(x0.shape[0])
    F_in = int(x0.shape[1])
    return T, N, F_in


def compute_alignment(
    mode: str,
    snapshots_dir: str,
    events_dir: str,
    train_ratio: float,
    val_ratio: float,
    truncate_icews_real_to: int,
    device: torch.device,
) -> AlignmentReport:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios: need train_ratio>0, val_ratio>=0, train_ratio+val_ratio<1")

    T, N, F_in = _infer_snapshot_config(snapshots_dir)

    ds = GC_TPP_Dataset(
        snapshots_dir=snapshots_dir,
        events_dir=events_dir,
        T=T,
        N=N,
        F_in=F_in,
        device=device,
        save_to_disk=True,
        mode=mode,
        truncate_icews_real_to=truncate_icews_real_to,
    )

    idx_tr, idx_va, idx_te, *_ = ds.get_train_val_test_split(train_ratio=train_ratio, val_ratio=val_ratio)

    # important: dataset expects tensors (can be cpu); this function returns per-split flags
    flags = ds.get_seen_ood_flags(idx_tr, idx_va, idx_te)

    seen_test = flags["seen_test"].detach().to("cpu")
    ood_test = flags["ood_test"].detach().to("cpu")

    if seen_test.numel() != idx_te.numel() or ood_test.numel() != idx_te.numel():
        raise RuntimeError(
            "Unexpected flags semantics: expected per-split flags with length == len(idx_test). "
            f"got seen_test={seen_test.numel()} ood_test={ood_test.numel()} idx_test={idx_te.numel()}"
        )

    # next-event positions: i predicts i+1, so positions 1..end are evaluated
    n_test_next = max(int(idx_te.numel()) - 1, 0)
    seen_next = int(seen_test[1:].sum().item()) if n_test_next > 0 else 0
    ood_next = int(ood_test[1:].sum().item()) if n_test_next > 0 else 0

    return AlignmentReport(
        mode=mode,
        snapshots_dir=snapshots_dir,
        events_dir=events_dir,
        T=int(T),
        N=int(N),
        F_in=int(F_in),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        n_total=int(ds.dt.numel()),
        n_train=int(idx_tr.numel()),
        n_val=int(idx_va.numel()),
        n_test=int(idx_te.numel()),
        n_test_next_event=int(n_test_next),
        seen_test_count=int(seen_test.sum().item()),
        ood_test_count=int(ood_test.sum().item()),
        seen_next_event_count=int(seen_next),
        ood_next_event_count=int(ood_next),
        seen_ood_key="(src, dst, coarse_type)",
        flags_semantics="per-split flags aligned with idx_test positions",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Check split/seen-ood alignment for GC-TPP datasets")
    ap.add_argument("--mode", default="icews_real_topk500_K500")
    ap.add_argument("--snapshots_dir", default="data/snapshots")
    ap.add_argument("--events_dir", default="data/events")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--truncate_icews_real_to", type=int, default=5000)
    ap.add_argument("--out", default=None, help="Optional path to write JSON report. Default: logs/alignment_{mode}_tr{train}_va{val}.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Only affects dataset tensors, not training.")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    rep = compute_alignment(
        mode=args.mode,
        snapshots_dir=args.snapshots_dir,
        events_dir=args.events_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        truncate_icews_real_to=args.truncate_icews_real_to,
        device=device,
    )

    # pretty print
    print("=" * 72)
    print("[ALIGNMENT REPORT]")
    print(f"mode = {rep.mode}")
    print(f"snapshots_dir = {rep.snapshots_dir}")
    print(f"events_dir    = {rep.events_dir}")
    print(f"snapshot config inferred: T={rep.T}, N={rep.N}, F_in={rep.F_in}")
    print("-")
    print(f"split ratios: train={rep.train_ratio:.3f}, val={rep.val_ratio:.3f}, test={1-rep.train_ratio-rep.val_ratio:.3f}")
    print(f"sizes: total={rep.n_total}, train={rep.n_train}, val={rep.n_val}, test={rep.n_test}")
    print(f"test next-event count (len(test)-1) = {rep.n_test_next_event}")
    print("-")
    print(f"Seen/OOD key = {rep.seen_ood_key}")
    print(f"flags semantics = {rep.flags_semantics}")
    print(f"test flags: seen={rep.seen_test_count}, ood={rep.ood_test_count} (len={rep.n_test})")
    print(f"next-event flags: seen_next={rep.seen_next_event_count}, ood_next={rep.ood_next_event_count} (len={rep.n_test_next_event})")
    print("=" * 72)

    out_path = args.out
    if out_path is None:
        os.makedirs("logs", exist_ok=True)
        # make filename stable
        tr = str(args.train_ratio).replace(".", "p")
        va = str(args.val_ratio).replace(".", "p")
        out_path = os.path.join("logs", f"alignment_{args.mode}_tr{tr}_va{va}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(rep), f, ensure_ascii=False, indent=2)
    print("[OK] wrote:", out_path)


if __name__ == "__main__":
    main()

