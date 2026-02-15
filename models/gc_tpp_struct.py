# models/gc_tpp_struct.py
"""
GC-TPP Struct model (Stage-5+11 early)

本文件在 GC-TPP Core 模型的基础上，构建节点级结构化版本：
- 复用 GraphEncoder / TimeEncoder；
- 在最后一张 snapshot 上，为每个节点 i 生成 [mu_i, log_sigma_i]；
- 对所有节点的 LogNormal NLL 取平均作为损失；
- 训练与评估流程与 Core 版本保持尽量一致，支持 toy / icews_real / icews_real_topk500 等 data_mode；
- 显式区分 Test-全体 / Test-Seen / Test-OOD 的 NLL / RMSE / MAE，
  并将曲线与指标保存到 logs/gc_tpp_struct_*.npz，作为 Stage-5+11(early)
  和后续 typed / Top-K / 主结果表的结构化基线。
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

from data.dataset_toy import GC_TPP_Dataset
from .gc_tpp_continuous import (
    GraphEncoder,
    TimeEncoder,
    EarlyStopping,
    lognormal_nll,
    set_seed,   # 新增：复用 Core 中的种子设置
)


# ===========================
# 1. GC-TPP-Struct 模型：节点 λ_i(t)
# ===========================
class GCTPPStruct(nn.Module):
    """
    在 GC-TPP-Core 的基础上：
    - 复用 GraphEncoder / TimeEncoder
    - 在最后一张 snapshot 上，为每个节点 i 生成 [mu_i, log_sigma_i]
    - 对所有节点的 NLL 取平均作为损失
    """

    def __init__(
        self,
        in_channels: int,
        graph_hidden_dim: int = 32,
        time_hidden_dim: int = 32,
        K: int = 3,
        max_history_len: int = 64,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.max_history_len = max_history_len

        self.graph_encoder = GraphEncoder(in_channels=in_channels, hidden_dim=graph_hidden_dim, K=K)
        self.time_encoder = TimeEncoder(hidden_dim=time_hidden_dim)

        fusion_dim = graph_hidden_dim + time_hidden_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_dim, 2),  # [mu_i, log_sigma_i]
        )

    def forward(self, X_snapshots: torch.Tensor, edge_index: torch.Tensor, dt_history: torch.Tensor):
        """
        X_snapshots: (T_snap, N, F_in)
        edge_index:  (2, E)
        dt_history:  (L_hist,)
        """
        g_t, H_all = self.graph_encoder(X_snapshots, edge_index, return_node_repr=True)
        H_last = H_all[-1]  # (N, graph_hidden_dim)

        h_t = self.time_encoder(dt_history)  # (time_hidden_dim,)
        h_rep = h_t.unsqueeze(0).expand(H_last.size(0), -1)  # (N, time_hidden_dim)

        z_nodes = torch.cat([H_last, h_rep], dim=-1)  # (N, fusion_dim)

        out = self.node_mlp(z_nodes)  # (N, 2)
        mu_nodes = out[:, 0]
        log_sigma_nodes = out[:, 1]

        lambda_nodes = torch.exp(mu_nodes + 0.5 * torch.exp(2 * log_sigma_nodes))
        # 额外返回 H_last，供 Typed‑proxy 使用
        return mu_nodes, log_sigma_nodes, lambda_nodes, H_last


# ===========================
# 2. 数据构建（含 Seen/OOD）
# ===========================
def build_events_from_dataset(
    device: torch.device,
    T_snap: int = 20,
    mode: str = "toy",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    ds = GC_TPP_Dataset(
        snapshots_dir="./data/snapshots",
        events_dir="./data/events",
        T=T_snap,
        N=10,
        F_in=3,
        device=device,
        mode=mode,
        save_to_disk=True,
    )

    (idx_train, idx_val, idx_test,
     event_times_train, event_times_val, event_times_test,
     dt_train, dt_val, dt_test) = ds.get_train_val_test_split(
        train_ratio=train_ratio, val_ratio=val_ratio
    )

    X_list = ds.X_list
    edge_index = ds.edge_index

    X_tensor = torch.stack(X_list, dim=0).to(device)
    edge_index = edge_index.to(device)

    return X_tensor, edge_index, event_times_train, dt_train, event_times_val, dt_val, event_times_test, dt_test


def build_events_from_dataset_with_flags(
    device: torch.device,
    T_snap: int = 20,
    mode: str = "toy",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    与原版唯一区别：在获得 train/val/test 索引后，通过
    GC_TPP_Dataset.get_seen_ood_flags(...) / get_triplets_split(...)
    生成 Seen/OOD 标记与标准化三元组切分，用于 Stage-5 和 typed-proxy debug。
    """
    ds = GC_TPP_Dataset(
        snapshots_dir="./data/snapshots",
        events_dir="./data/events",
        T=T_snap,
        N=10,
        F_in=3,
        device=device,
        mode=mode,
        save_to_disk=True,
    )

    (idx_train, idx_val, idx_test,
     event_times_train, event_times_val, event_times_test,
     dt_train, dt_val, dt_test) = ds.get_train_val_test_split(
        train_ratio=train_ratio, val_ratio=val_ratio
    )

    X_list = ds.X_list
    edge_index = ds.edge_index

    flags = ds.get_seen_ood_flags(idx_train, idx_val, idx_test)
    triplets = ds.get_triplets_split(idx_train, idx_val, idx_test)

    X_tensor = torch.stack(X_list, dim=0).to(device)
    edge_index = edge_index.to(device)

    return (
        X_tensor,
        edge_index,
        event_times_train, dt_train,
        event_times_val,   dt_val,
        event_times_test,  dt_test,
        flags,
        triplets,
    )


# ===========================
# 3. 训练 & 验证 & 测试 主流程（Struct）
# ===========================
def run_gc_tpp_struct(data_mode: str = "toy"):
    """
    训练并评估 GC-TPP Struct 模型的主入口（Stage-5+11 early）。
    """
    set_seed(0)  # 与 Core 使用同一个随机种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_history_len = 64
    T_snap = 20

    train_ratio = 0.7
    val_ratio = 0.15

    mode_tag = "toy" if data_mode == "toy" else data_mode

    (X_snapshots,
     edge_index,
     event_times_train,
     dt_train,
     event_times_val,
     dt_val,
     event_times_test,
     dt_test,
     flags, triplets) = build_events_from_dataset_with_flags(
        device=device, T_snap=T_snap, mode=data_mode, train_ratio=train_ratio, val_ratio=val_ratio
    )

    in_channels = X_snapshots.size(-1)
    model = GCTPPStruct(
        in_channels=in_channels,
        graph_hidden_dim=32,
        time_hidden_dim=32,
        K=3,
        max_history_len=max_history_len,
        dropout_prob=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stopper = EarlyStopping(patience=3, min_delta=1e-3)

    num_epochs = 15

    train_nll_list = []
    val_nll_list = []
    train_rmse_list = []
    val_rmse_list = []
    train_mae_list = []
    val_mae_list = []

    print(f"[INFO] Using device: {device}")
    print(
        "[INFO] Train/Val/Test sizes (dt) = {} / {} / {}".format(
            int(dt_train.numel()),
            int(dt_val.numel()),
            int(dt_test.numel()),
        )
    )

    src_train = dst_train = type_train = None
    src_val = dst_val = type_val = None
    src_test = dst_test = type_test = None
    trip_reason = None
    if isinstance(triplets, dict):
        trip_reason = triplets.get("reason")
        if trip_reason != "ok":
            print(f"[WARN] Triplets split unavailable -> reason={trip_reason}")
        else:
            src_train, dst_train, type_train = triplets["src_train"], triplets["dst_train"], triplets["type_train"]
            src_val,   dst_val,   type_val   = triplets["src_val"],   triplets["dst_val"],   triplets["type_val"]
            src_test,  dst_test,  type_test  = triplets["src_test"],  triplets["dst_test"],  triplets["type_test"]

    # ---------- Train + Val ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_total_nll = 0.0
        train_se_sum = 0.0
        train_ae_sum = 0.0
        train_count = 0
        debug_lambda = None

        for i in range(dt_train.numel() - 1):
            optimizer.zero_grad()

            start_idx = max(0, i - max_history_len + 1)
            dt_history = dt_train[start_idx: i + 1]
            dt_next = dt_train[i + 1]

            mu_nodes, log_sigma_nodes, lambda_nodes, _ = model(X_snapshots, edge_index, dt_history)

            nll_nodes = lognormal_nll(dt_next, mu_nodes, log_sigma_nodes)
            nll = nll_nodes.mean()

            log_dt_true = torch.log(torch.clamp(dt_next, min=1e-8))
            log_dt_pred_nodes = mu_nodes.detach()
            err_log_nodes = log_dt_pred_nodes - log_dt_true

            se = float(torch.mean(err_log_nodes ** 2))
            ae = float(torch.mean(torch.abs(err_log_nodes)))

            train_se_sum += se
            train_ae_sum += ae

            nll.backward()
            optimizer.step()

            train_total_nll += float(nll.item())
            train_count += 1
            debug_lambda = lambda_nodes

        avg_train_nll = train_total_nll / max(train_count, 1)
        train_rmse = math.sqrt(train_se_sum / max(train_count, 1))
        train_mae = train_ae_sum / max(train_count, 1)

        train_nll_list.append(avg_train_nll)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)

        model.eval()
        val_total_nll = 0.0
        val_se_sum = 0.0
        val_ae_sum = 0.0
        val_count = 0
        debug_lambda_val = None

        with torch.no_grad():
            for i in range(dt_val.numel() - 1):
                start_idx = max(0, i - max_history_len + 1)
                dt_history = dt_val[start_idx: i + 1]
                dt_next = dt_val[i + 1]

                mu_nodes, log_sigma_nodes, lambda_nodes, _ = model(X_snapshots, edge_index, dt_history)

                nll_nodes = lognormal_nll(dt_next, mu_nodes, log_sigma_nodes)
                nll = nll_nodes.mean()

                log_dt_true = torch.log(torch.clamp(dt_next, min=1e-8))
                log_dt_pred_nodes = mu_nodes
                err_log_nodes = log_dt_pred_nodes - log_dt_true

                se = float(torch.mean(err_log_nodes ** 2))
                ae = float(torch.mean(torch.abs(err_log_nodes)))

                val_se_sum += se
                val_ae_sum += ae

                val_total_nll += float(nll.item())
                val_count += 1
                debug_lambda_val = lambda_nodes

        avg_val_nll = val_total_nll / max(val_count, 1)
        val_rmse = math.sqrt(val_se_sum / max(val_count, 1))
        val_mae = val_ae_sum / max(val_count, 1)

        val_nll_list.append(avg_val_nll)
        val_rmse_list.append(val_rmse)
        val_mae_list.append(val_mae)

        print(
            f"Epoch {epoch:02d} | GC-TPP-Struct ({mode_tag}) "
            f"Train NLL = {avg_train_nll:.4f} | Val NLL = {avg_val_nll:.4f} "
            f"| Train RMSE (log Δt) = {train_rmse:.4f} | Val RMSE (log Δt) = {val_rmse:.4f} "
            f"| Train MAE (log Δt) = {train_mae:.4f} | Val MAE (log Δt) = {val_mae:.4f}"
        )

        if debug_lambda_val is not None:
            lam_np = debug_lambda_val.detach().cpu().numpy()
            print(f"    [DEBUG] 示例节点 λ_i(t) (前 5 个) = {lam_np[:5]}")

        scheduler.step(avg_val_nll)
        if early_stopper.step(avg_val_nll):
            print(f"[INFO] Early stopping triggered at epoch {epoch}. Best Val NLL = {early_stopper.best:.4f}")
            break

    # ---------- Test (Seen / OOD) ----------
    model.eval()

    test_total_nll = 0.0
    test_se_sum = 0.0
    test_ae_sum = 0.0
    test_count = 0

    test_total_nll_seen = 0.0
    test_se_sum_seen = 0.0
    test_ae_sum_seen = 0.0
    test_count_seen = 0

    test_total_nll_ood = 0.0
    test_se_sum_ood = 0.0
    test_ae_sum_ood = 0.0
    test_count_ood = 0

    seen_test = flags["seen_test"] if isinstance(flags, dict) and "seen_test" in flags else None
    ood_test = flags["ood_test"] if isinstance(flags, dict) and "ood_test" in flags else None

    try:
        n_dt = int(dt_test.numel()) if hasattr(dt_test, "numel") else None
        n_seen = int(seen_test.numel()) if (seen_test is not None and hasattr(seen_test, "numel")) else None
        n_ood = int(ood_test.numel()) if (ood_test is not None and hasattr(ood_test, "numel")) else None

        seen_cnt = int(seen_test.sum().item()) if (seen_test is not None and hasattr(seen_test, "sum")) else None
        ood_cnt = int(ood_test.sum().item()) if (ood_test is not None and hasattr(ood_test, "sum")) else None

        print(
            f"[INFO] Stage-5 flags (struct): "
            f"dt_test_len={n_dt} | seen_test_len={n_seen}, seen_sum={seen_cnt} "
            f"| ood_test_len={n_ood}, ood_sum={ood_cnt}"
        )
    except Exception:
        pass

    with torch.no_grad():
        for i in range(dt_test.numel() - 1):
            start_idx = max(0, i - max_history_len + 1)
            dt_history = dt_test[start_idx: i + 1]
            dt_next = dt_test[i + 1]

            mu_nodes, log_sigma_nodes, lambda_nodes, _ = model(X_snapshots, edge_index, dt_history)

            nll_nodes = lognormal_nll(dt_next, mu_nodes, log_sigma_nodes)
            nll = nll_nodes.mean()

            log_dt_true = torch.log(torch.clamp(dt_next, min=1e-8))
            log_dt_pred_nodes = mu_nodes
            err_log_nodes = log_dt_pred_nodes - log_dt_true

            se = float(torch.mean(err_log_nodes ** 2))
            ae = float(torch.mean(torch.abs(err_log_nodes)))

            test_se_sum += se
            test_ae_sum += ae
            test_total_nll += float(nll.item())
            test_count += 1

            ev_idx = i + 1
            is_seen = False
            is_ood = False
            if seen_test is not None and ev_idx < int(seen_test.numel()):
                is_seen = bool(seen_test[ev_idx].item())
            if ood_test is not None and ev_idx < int(ood_test.numel()):
                is_ood = bool(ood_test[ev_idx].item())

            if is_seen:
                test_se_sum_seen += se
                test_ae_sum_seen += ae
                test_total_nll_seen += float(nll.item())
                test_count_seen += 1

            if is_ood:
                test_se_sum_ood += se
                test_ae_sum_ood += ae
                test_total_nll_ood += float(nll.item())
                test_count_ood += 1

    def _safe_mean(total, count):
        return total / count if count > 0 else float("nan")

    avg_test_nll = _safe_mean(test_total_nll, test_count)
    test_rmse = math.sqrt(_safe_mean(test_se_sum, test_count)) if test_count > 0 else float("nan")
    test_mae = _safe_mean(test_ae_sum, test_count)

    avg_test_nll_seen = _safe_mean(test_total_nll_seen, test_count_seen)
    test_rmse_seen = math.sqrt(_safe_mean(test_se_sum_seen, test_count_seen)) if test_count_seen > 0 else float("nan")
    test_mae_seen = _safe_mean(test_ae_sum_seen, test_count_seen)

    avg_test_nll_ood = _safe_mean(test_total_nll_ood, test_count_ood)
    test_rmse_ood = math.sqrt(_safe_mean(test_se_sum_ood, test_count_ood)) if test_count_ood > 0 else float("nan")
    test_mae_ood = _safe_mean(test_ae_sum_ood, test_count_ood)

    print(f"[INFO] Test NLL  (Struct {mode_tag}, all) = {avg_test_nll:.4f}")
    print(f"[INFO] Test RMSE (log Δt, Struct {mode_tag}, all) = {test_rmse:.4f}")
    print(f"[INFO] Test MAE  (log Δt, Struct {mode_tag}, all) = {test_mae:.4f}")

    print(f"[INFO] Test NLL  (Struct {mode_tag}, Seen) = {avg_test_nll_seen:.4f} | count={test_count_seen}")
    print(f"[INFO] Test RMSE (log Δt, Struct {mode_tag}, Seen) = {test_rmse_seen:.4f} | count={test_count_seen}")
    print(f"[INFO] Test MAE  (log Δt, Struct {mode_tag}, Seen) = {test_mae_seen:.4f} | count={test_count_seen}")

    print(f"[INFO] Test NLL  (Struct {mode_tag}, OOD)  = {avg_test_nll_ood:.4f} | count={test_count_ood}")
    print(f"[INFO] Test RMSE (log Δt, Struct {mode_tag}, OOD)  = {test_rmse_ood:.4f} | count={test_count_ood}")
    print(f"[INFO] Test MAE  (log Δt, Struct {mode_tag}, OOD)  = {test_mae_ood:.4f} | count={test_count_ood}")

    os.makedirs("logs", exist_ok=True)
    if data_mode == "toy":
        save_path = "logs/gc_tpp_struct_toy.npz"
    elif data_mode == "icews_real":
        save_path = "logs/gc_tpp_struct_icews_real.npz"
    else:
        save_path = f"logs/gc_tpp_struct_{data_mode}.npz"

    np.savez(
        save_path,
        train_nll=np.array(train_nll_list),
        val_nll=np.array(val_nll_list),
        train_rmse=np.array(train_rmse_list),
        val_rmse=np.array(val_rmse_list),
        train_mae=np.array(train_mae_list),
        val_mae=np.array(val_mae_list),
        test_nll=avg_test_nll,
        test_rmse=test_rmse,
        test_mae=test_mae,
        test_nll_seen=avg_test_nll_seen,
        test_rmse_seen=test_rmse_seen,
        test_mae_seen=test_mae_seen,
        test_count_seen=test_count_seen,
        test_nll_ood=avg_test_nll_ood,
        test_rmse_ood=test_rmse_ood,
        test_mae_ood=test_mae_ood,
        test_count_ood=test_count_ood,
    )
    print(f"[INFO] Saved Struct curves to {save_path}")

    results = {
        "data_mode": data_mode,
        "train_nll": train_nll_list,
        "val_nll": val_nll_list,
        "train_rmse": train_rmse_list,
        "val_rmse": val_rmse_list,
        "train_mae": train_mae_list,
        "val_mae": val_mae_list,
        "test_nll": avg_test_nll,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_nll_seen": avg_test_nll_seen,
        "test_rmse_seen": test_rmse_seen,
        "test_mae_seen": test_mae_seen,
        "test_count_seen": test_count_seen,
        "test_nll_ood": avg_test_nll_ood,
        "test_rmse_ood": test_rmse_ood,
        "test_mae_ood": test_mae_ood,
        "test_count_ood": test_count_ood,
        "log_path": save_path,
    }
    return results
