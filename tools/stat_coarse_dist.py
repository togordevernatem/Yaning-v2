# tools/stat_coarse_dist.py
import os
from typing import Dict, Any, List

import pandas as pd
import torch

from data.dataset_toy import GC_TPP_Dataset
from data.event_type_mapping import COARSE_LABELS


def stat_coarse_distribution(
    mode: str = "icews_real_topk500_K500",
    snapshots_dir: str = "./data/snapshots",
    events_dir: str = "./data/events",
    out_csv: str = "logs/coarse_distribution_icews_real_topk500_K500.csv",
):
    """
    统计给定 data_mode 下 coarse 类型在 train/val/test、Seen/OOD 上的分布。

    输出字段：
    - split: train/val/test
    - coarse_id, coarse_name
    - count_total
    - count_seen
    - count_ood
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GC_TPP_Dataset(
        snapshots_dir=snapshots_dir,
        events_dir=events_dir,
        T=20,
        N=10,
        F_in=3,
        device=device,
        mode=mode,
        save_to_disk=True,
    )

    (
        idx_train,
        idx_val,
        idx_test,
        ev_time_train,
        ev_time_val,
        ev_time_test,
        dt_train,
        dt_val,
        dt_test,
    ) = ds.get_train_val_test_split(train_ratio=0.7, val_ratio=0.15)

    flags = ds.get_seen_ood_flags(idx_train, idx_val, idx_test)
    coarse_all = ds.get_event_coarse_types()  # [num_events]

    rows: List[Dict[str, Any]] = []

    def collect(split_name: str, idx: torch.Tensor, seen_flag: torch.Tensor, ood_flag: torch.Tensor):
        idx = idx.to(coarse_all.device)
        c_split = coarse_all[idx]

        seen_flag = seen_flag.to(coarse_all.device)
        ood_flag = ood_flag.to(coarse_all.device)

        for cid in torch.unique(c_split):
            cid_int = int(cid.item())
            name = COARSE_LABELS.get(cid_int, "Other")

            mask = (c_split == cid)

            count_total = int(mask.sum().item())
            count_seen = int(seen_flag[mask].sum().item())
            count_ood = int(ood_flag[mask].sum().item())

            rows.append({
                "split": split_name,
                "coarse_id": cid_int,
                "coarse_name": name,
                "count_total": count_total,
                "count_seen": count_seen,
                "count_ood": count_ood,
            })

    collect("train", idx_train, flags["seen_train"], flags["ood_train"])
    collect("val", idx_val, flags["seen_val"], flags["ood_val"])
    collect("test", idx_test, flags["seen_test"], flags["ood_test"])

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[COARSE-STAT] Saved coarse distribution to {out_csv}")
    print(df)


if __name__ == "__main__":
    stat_coarse_distribution(
        mode="icews_real_topk500_K500",
        snapshots_dir="./data/snapshots",
        events_dir="./data/events",
        out_csv="logs/coarse_distribution_icews_real_topk500_K500.csv",
    )
