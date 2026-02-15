import os
import numpy as np
import torch

from data.dataset_toy import GC_TPP_Dataset
from data.event_type_mapping import COARSE_LABELS
from models.gc_tpp_struct import build_events_from_dataset_with_flags
from models.gc_tpp_continuous import set_seed


def build_node_profile(
    data_mode: str = "icews_real_topk500_K500",
    T_snap: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    save_dir: str = "./data",
):
    """
    根据 train 部分的事件，为每个节点构建 coarse 分布 profile:
      profile[i, c] = 该节点在 train 中作为 src/dst 参与 coarse=c 的次数 / 总次数（加 eps）
    只用 train 事件，防止信息泄漏。
    保存为: {save_dir}/node_profile_{data_mode}.npy, shape = (N, C)
    """
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        X_snapshots,
        edge_index,
        event_times_train,
        dt_train,
        event_times_val,
        dt_val,
        event_times_test,
        dt_test,
        flags,
        triplets,
    ) = build_events_from_dataset_with_flags(
        device=device,
        T_snap=T_snap,
        mode=data_mode,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    N = X_snapshots.size(1)
    C = len(COARSE_LABELS)

    # 用 GC_TPP_Dataset 再拿一次 coarse_all (所有事件的 coarse_id)
    ds_for_coarse = GC_TPP_Dataset(
        snapshots_dir="./data/snapshots",
        events_dir="./data/events",
        T=T_snap,
        N=N,
        F_in=X_snapshots.size(-1),
        device=device,
        mode=data_mode,
        save_to_disk=True,
    )
    coarse_all = ds_for_coarse.get_event_coarse_types().to(device)  # [num_events]

    if not isinstance(triplets, dict) or triplets.get("reason") != "ok":
        raise RuntimeError(f"Triplets not available or invalid: {triplets}")

    src_train = triplets["src_train"].to(device)  # [num_train_events]
    dst_train = triplets["dst_train"].to(device)

    node_coarse_counts = torch.zeros(N, C, device=device)

    num_train_events = int(src_train.numel())
    # 按你当前代码风格，ev_idx = i+1 对齐 coarse_all
    for ev_idx in range(num_train_events):
        if ev_idx + 1 >= int(coarse_all.numel()):
            break
        c_idx = int(coarse_all[ev_idx + 1].item())
        if c_idx < 0 or c_idx >= C:
            continue

        s = int(src_train[ev_idx].item())
        d = int(dst_train[ev_idx].item())
        if 0 <= s < N:
            node_coarse_counts[s, c_idx] += 1.0
        if 0 <= d < N:
            node_coarse_counts[d, c_idx] += 1.0

    # 归一化为分布
    eps = 1e-8
    sums = node_coarse_counts.sum(dim=1, keepdim=True)  # (N, 1)
    node_profile = node_coarse_counts / (sums + eps)   # (N, C)

    node_profile_np = node_profile.detach().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"node_profile_{data_mode}.npy")
    np.save(out_path, node_profile_np)
    print(f"[Profile] Saved node_profile to {out_path} with shape {node_profile_np.shape}")


if __name__ == "__main__":
    build_node_profile()
