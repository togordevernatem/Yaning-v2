# tools/build_node_profile.py
"""
Offline builder for node-level coarse profiles (train-only, no leakage).

Usage (from repo root):

    python -m tools.build_node_profile --data_mode icews_real_topk500_K500

This script will:
1) Load GC_TPP_Dataset and build events with flags/triplets.
2) Only use TRAIN events (src_train / dst_train + coarse type) to count per-node coarse frequencies.
3) L1-normalize per node (row-wise), with epsilon to avoid log issues.
4) Keep all-zero rows for nodes never seen in train (safe for cold-start).
5) Save node_profile_{data_mode}.npy under ./data/

The resulting profile is then consumed by models/gc_tpp_struct_typed.py.
"""

import argparse
import os

import numpy as np
import torch

from data.dataset_toy import GC_TPP_Dataset
from models.gc_tpp_struct import build_events_from_dataset_with_flags, set_seed


def build_node_profile(data_mode: str, T_snap: int = 20, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Build node-level coarse-type profiles for a given data_mode.

    Returns:
        profile: np.ndarray of shape (N, C), where
                 N = number of nodes,
                 C = number of coarse types.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    print("============================================================")
    print(f"[PROFILE-BUILDER] data_mode = {data_mode}")
    print(f"[PROFILE-BUILDER] device    = {device}")
    print("============================================================")

    # 1) Load dataset to infer N and coarse_all
    ds = GC_TPP_Dataset(
        snapshots_dir="./data/snapshots",
        events_dir="./data/events",
        T=T_snap,
        N=None,            # let dataset infer from snapshots
        F_in=None,         # let dataset infer from snapshots
        device=device,
        mode=data_mode,
        save_to_disk=True,
    )

    # coarse_all: (num_events,) with coarse id per event
    coarse_all = ds.get_event_coarse_types().to(device)
    num_events = int(coarse_all.numel())
    print(f"[PROFILE-BUILDER] Loaded coarse_all with {num_events} events.")

    # 2) Build events with flags/triplets to get train split and nodes
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

    # shapes
    T_snap_actual = X_snapshots.size(0)
    N = X_snapshots.size(1)
    print(f"[PROFILE-BUILDER] X_snapshots shape = {X_snapshots.shape} (T_snap={T_snap_actual}, N={N}, F_in={X_snapshots.size(-1)})")

    # triplets: dict with "train"/"val"/"test", each being dict of src_idx, dst_idx, type_idx, time_idx
    if not isinstance(triplets, dict) or "train" not in triplets:
        raise RuntimeError("[PROFILE-BUILDER] triplets['train'] not found, cannot build profile.")

    train_triplets = triplets["train"]
    src_train = train_triplets["src_idx"].to(device)   # (num_train,)
    dst_train = train_triplets["dst_idx"].to(device)   # (num_train,)
    type_train = train_triplets["type_idx"].to(device) # (num_train,)

    num_train = int(src_train.numel())
    print(f"[PROFILE-BUILDER] num_train events = {num_train}")

    if num_train == 0:
        raise RuntimeError("[PROFILE-BUILDER] No train events found, cannot build profile.")

    # 3) Determine number of coarse types
    max_coarse_id = int(torch.max(coarse_all).item())
    C = max_coarse_id + 1
    print(f"[PROFILE-BUILDER] Num coarse types (C) = {C} (max coarse id = {max_coarse_id})")

    # 4) Count per-node coarse frequencies from TRAIN events only
    # Initialize N x C matrix
    profile = torch.zeros((N, C), dtype=torch.float32, device=device)

    # For each train event (src, dst, type_idx), we map type_idx -> coarse_id via coarse_all
    # Assumption: coarse_all and type_idx share the same indexing for events
    # If your dataset ties type_idx directly to coarse_all[time_idx], adjust accordingly.
    if "time_idx" in train_triplets:
        # Preferred: use explicit time_idx to index coarse_all
        time_train = train_triplets["time_idx"].to(device)  # (num_train,)
        # Just to be safe, clamp in range
        time_train_clamped = torch.clamp(time_train, 0, coarse_all.numel() - 1)
        coarse_train = coarse_all[time_train_clamped]  # (num_train,)
    else:
        # Fallback: assume type_idx aligns with coarse_all indices (only safe for toy/simplified)
        print("[PROFILE-BUILDER][WARN] 'time_idx' not found in train_triplets, "
              "fallback to using type_idx as coarse index. "
              "Please verify this assumption for your dataset.")
        type_train_clamped = torch.clamp(type_train, 0, coarse_all.numel() - 1)
        coarse_train = coarse_all[type_train_clamped]

    # Now coarse_train: (num_train,) coarse id for each train event
    # We update profile[src, coarse], profile[dst, coarse]
    for i in range(num_train):
        c = int(coarse_train[i].item())
        s = int(src_train[i].item())
        d = int(dst_train[i].item())
        if 0 <= s < N:
            profile[s, c] += 1.0
        if 0 <= d < N:
            profile[d, c] += 1.0

    # 5) Row-wise L1 normalization with epsilon, keep all-zero rows
    eps = 1e-8
    row_sums = profile.sum(dim=1, keepdim=True)  # (N, 1)
    # Avoid division by zero: only normalize rows with positive sum
    nonzero_mask = row_sums.squeeze(1) > 0
    profile_norm = profile.clone()
    profile_norm[nonzero_mask] = profile[nonzero_mask] / (row_sums[nonzero_mask] + eps)
    # Zero rows (never seen in train) remain all zero

    # 6) Save to npy
    os.makedirs("./data", exist_ok=True)
    save_path = os.path.join("./data", f"node_profile_{data_mode}.npy")
    np.save(save_path, profile_norm.cpu().numpy())
    print(f"[PROFILE-BUILDER] Saved node profile to {save_path} with shape {tuple(profile_norm.shape)}")

    return profile_norm


def main():
    parser = argparse.ArgumentParser(description="Build node-level coarse profiles for GC-TPP.")
    parser.add_argument(
        "--data_mode",
        type=str,
        default="icews_real_topk500_K500",
        help=(
            "数据模式，例如：toy / icews_real / icews_real_topk500 / "
            "icews_real_topk500_K100 / icews_real_topk500_K500 / icews_real_topk500_K1000"
        ),
    )
    parser.add_argument(
        "--T_snap",
        type=int,
        default=20,
        help="Number of snapshots T used in build_events_from_dataset_with_flags (keep consistent with training).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train ratio used in build_events_from_dataset_with_flags.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Val ratio used in build_events_from_dataset_with_flags.",
    )
    args = parser.parse_args()

    build_node_profile(
        data_mode=args.data_mode,
        T_snap=args.T_snap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()