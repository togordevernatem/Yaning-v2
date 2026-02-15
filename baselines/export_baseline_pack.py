import os, argparse, json
import numpy as np
import torch
from data.dataset_toy import GC_TPP_Dataset

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mode", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--truncate", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 和你仓库主线一致：只用来读取 event stream + dt + flags，不训练
    ds = GC_TPP_Dataset(
        snapshots_dir="data/snapshots",
        events_dir="data/events",
        T=20, N=10, F_in=3,
        device=torch.device("cpu"),
        save_to_disk=True,
        mode=args.data_mode,
        truncate_icews_real_to=args.truncate,
    )

    # split（注意：你当前 dataset_toy 的 split 是确定性的，所以 seed 不影响这一层）
    idx_tr, idx_va, idx_te, t_tr, t_va, t_te, dt_tr, dt_va, dt_te = ds.get_train_val_test_split(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    flags = ds.get_seen_ood_flags(idx_tr, idx_va, idx_te)
    # flags keys: seen_train/ood_train/seen_val/ood_val/seen_test/ood_test

    pack = {
        "data_mode": args.data_mode,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "truncate": args.truncate,
        "n_total": int(len(ds.dt)),
        "n_train": int(len(idx_tr)),
        "n_val": int(len(idx_va)),
        "n_test": int(len(idx_te)),
        "n_test_seen": int(to_np(flags["seen_test"]).sum()),
        "n_test_ood": int(to_np(flags["ood_test"]).sum()),
    }

    out_npz = os.path.join(args.out_dir, "baseline_pack.npz")
    np.savez(
        out_npz,
        # meta
        data_mode=np.array([args.data_mode]),
        train_ratio=np.array([args.train_ratio], dtype=np.float32),
        val_ratio=np.array([args.val_ratio], dtype=np.float32),
        truncate=np.array([args.truncate], dtype=np.int64),

        # full stream
        src=to_np(ds.src).astype(np.int64),
        dst=to_np(ds.dst).astype(np.int64),
        ev_type=to_np(ds.ev_type).astype(np.int64),
        event_times=to_np(ds.event_times).astype(np.float64),
        dt=to_np(ds.dt).astype(np.float64),

        # split indices
        idx_train=to_np(idx_tr).astype(np.int64),
        idx_val=to_np(idx_va).astype(np.int64),
        idx_test=to_np(idx_te).astype(np.int64),

        # split arrays (time / dt)
        event_times_train=to_np(t_tr).astype(np.float64),
        event_times_val=to_np(t_va).astype(np.float64),
        event_times_test=to_np(t_te).astype(np.float64),
        dt_train=to_np(dt_tr).astype(np.float64),
        dt_val=to_np(dt_va).astype(np.float64),
        dt_test=to_np(dt_te).astype(np.float64),

        # flags (bool)
        seen_train=to_np(flags["seen_train"]).astype(np.bool_),
        ood_train=to_np(flags["ood_train"]).astype(np.bool_),
        seen_val=to_np(flags["seen_val"]).astype(np.bool_),
        ood_val=to_np(flags["ood_val"]).astype(np.bool_),
        seen_test=to_np(flags["seen_test"]).astype(np.bool_),
        ood_test=to_np(flags["ood_test"]).astype(np.bool_),
    )

    out_json = os.path.join(args.out_dir, "baseline_pack_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    print("[OK] exported baseline pack ->", out_npz)
    print("[OK] meta ->", out_json)
    print("     sizes: train/val/test =", pack["n_train"], pack["n_val"], pack["n_test"])
    print("     test  : seen/ood      =", pack["n_test_seen"], pack["n_test_ood"])

if __name__ == "__main__":
    main()
