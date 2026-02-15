import os, json, argparse
import numpy as np
import torch

from data.dataset_toy import GC_TPP_Dataset

def to_np(x, dtype=None):
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mode", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--truncate", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = GC_TPP_Dataset(
        snapshots_dir="data/snapshots",
        events_dir="data/events",
        T=20, N=10, F_in=3,
        device=torch.device("cpu"),
        save_to_disk=False,
        mode=args.data_mode,
        truncate_icews_real_to=args.truncate,
    )

    idx_tr, idx_va, idx_te, t_tr, t_va, t_te, dt_tr, dt_va, dt_te = ds.get_train_val_test_split(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    flags = ds.get_seen_ood_flags(
        idx_train=idx_tr.to(ds.device),
        idx_val=idx_va.to(ds.device),
        idx_test=idx_te.to(ds.device),
    )

    # marks (coarse types) for each split
    trip = ds.get_triplets_split(
        idx_train=idx_tr.to(ds.device),
        idx_val=idx_va.to(ds.device),
        idx_test=idx_te.to(ds.device),
    )
    marks_tr = trip["coarse_train"]
    marks_va = trip["coarse_val"]
    marks_te = trip["coarse_test"]

    pack_path = os.path.join(args.out_dir, "baseline_pack.npz")
    np.savez(
        pack_path,
        idx_train=to_np(idx_tr, np.int64),
        idx_val=to_np(idx_va, np.int64),
        idx_test=to_np(idx_te, np.int64),
        event_time_train=to_np(t_tr, np.float64),
        event_time_val=to_np(t_va, np.float64),
        event_time_test=to_np(t_te, np.float64),
        dt_train=to_np(dt_tr, np.float64),
        dt_val=to_np(dt_va, np.float64),
        dt_test=to_np(dt_te, np.float64),
        seen_train=to_np(flags["seen_train"], np.bool_),
        seen_val=to_np(flags["seen_val"], np.bool_),
        seen_test=to_np(flags["seen_test"], np.bool_),
        ood_train=to_np(flags["ood_train"], np.bool_),
        ood_val=to_np(flags["ood_val"], np.bool_),
        ood_test=to_np(flags["ood_test"], np.bool_),
        mark_train=to_np(marks_tr, np.int64),
        mark_val=to_np(marks_va, np.int64),
        mark_test=to_np(marks_te, np.int64),
    )

    meta = {
        "data_mode": args.data_mode,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "truncate": args.truncate,
        "sizes": {
            "train": int(len(idx_tr)),
            "val": int(len(idx_va)),
            "test": int(len(idx_te)),
            "test_seen": int(to_np(flags["seen_test"]).sum()),
            "test_ood": int(to_np(flags["ood_test"]).sum()),
        },
        "note": "Baseline pack exported from GC_TPP_Dataset splits + coarse marks.",
    }
    meta_path = os.path.join(args.out_dir, "baseline_pack.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", pack_path)
    print("[OK] meta :", meta_path)
    print("[SIZES] train/val/test =", meta["sizes"]["train"], meta["sizes"]["val"], meta["sizes"]["test"],
          "| test seen/ood =", meta["sizes"]["test_seen"], meta["sizes"]["test_ood"])

if __name__ == "__main__":
    main()
