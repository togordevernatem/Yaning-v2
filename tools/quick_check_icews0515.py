"""tools/quick_check_icews0515.py

最小冒烟检查：确保 mode=icews0515 的 GC_TPP_Dataset 可以顺利初始化，
并输出一些关键统计，作为“可以删旧数据”的护栏。

通过标准：
- 能实例化 dataset
- event_times/dt/src/dst/ev_type 维度一致且 >0
- dt 全部为正（适配 log(dt) 回归）

用法：
  python -m tools.quick_check_icews0515
  python -m tools.quick_check_icews0515 --mode icews0515 --events_dir ./data/events
"""

from __future__ import annotations

import argparse
import torch

from data.dataset_toy import GC_TPP_Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="icews0515")
    ap.add_argument("--events_dir", type=str, default="./data/events")
    ap.add_argument("--snapshots_dir", type=str, default="./data/snapshots")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--F_in", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GC_TPP_Dataset(
        snapshots_dir=args.snapshots_dir,
        events_dir=args.events_dir,
        T=args.T,
        N=args.N,
        F_in=args.F_in,
        device=device,
        mode=args.mode,
        save_to_disk=True,
    )

    E = ds.event_times.numel()
    print("[quick_check] mode=", args.mode)
    print("[quick_check] E=", E)
    print("[quick_check] event_times.shape=", tuple(ds.event_times.shape))
    print("[quick_check] dt.shape=", tuple(ds.dt.shape))
    print("[quick_check] src/dst/ev_type.shape=", tuple(ds.src.shape), tuple(ds.dst.shape), tuple(ds.ev_type.shape))

    assert E > 0, "No events loaded"
    assert ds.dt.numel() == E and ds.src.numel() == E and ds.dst.numel() == E and ds.ev_type.numel() == E

    num_nonpos = int((ds.dt <= 0).sum().item())
    print("[quick_check] dt<=0 count =", num_nonpos)
    assert num_nonpos == 0, "dt contains non-positive entries; log(dt) would break"

    # Seen/OOD 逻辑也跑一下，确保不炸
    idx_train, idx_val, idx_test, *_ = ds.get_train_val_test_split(train_ratio=0.7, val_ratio=0.15)
    flags = ds.get_seen_ood_flags(idx_train, idx_val, idx_test)
    print("[quick_check] split sizes:", idx_train.numel(), idx_val.numel(), idx_test.numel())
    print("[quick_check] seen_test count:", int(flags["seen_test"].sum().item()))
    print("[quick_check] ood_test  count:", int(flags["ood_test"].sum().item()))

    print("✅ quick_check passed")


if __name__ == "__main__":
    main()

