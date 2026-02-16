# models/gc_tpp_struct_typed.py
"""
GC-TPP Struct-Typed model (Profile-Augmented, NO Plan C/D residual)

核心设计：
1. 结构级 Profile 注入（杀手锏）：
   - 预先用 tools/build_node_profile.py 按 train 事件统计每个节点的 coarse 分布 profile[i, c]。
   - 在构建 X_snapshots 后，将 profile 拼接为额外特征维度，进入 GraphEncoder。
2. 不再对 mu/log_sigma 做任何 Typed 残差（去掉 Plan C/D）：
   - 训练 / 验证 / 测试阶段，NLL 只用 backbone 输出的 mu_nodes, log_sigma_nodes 计算。
   - Typed 头暂时只作为分析接口存在（不改时间分布）。
3. Table1/Table2 自动落盘：
   - Test 结束后写出 metrics.json（All/Seen/OOD 的 NLL/RMSE/MAE + Coverage@90）
   - 并写出 OOD-only 的 per-coarse 分组（供 Table2 使用）
"""

import os
import math
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn

from data.dataset_toy import GC_TPP_Dataset
from data.event_type_mapping import COARSE_LABELS
from models.gc_tpp_struct import (
    build_events_from_dataset_with_flags,
    lognormal_nll,
    set_seed,
    GraphEncoder,
    TimeEncoder,
    EarlyStopping,
)


# ===========================
# 1. GC-TPP-Struct Backbone
# ===========================
class GCTPPStructBackbone(nn.Module):
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
        X_snapshots: (T_snap, N, F_in_aug)  # 已含 profile
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
        return mu_nodes, log_sigma_nodes, lambda_nodes, H_last


class GCTPPStructTyped(nn.Module):
    """
    目前 Typed 头只作为分析/扩展接口，不参与 NLL 计算。
    后续如果要做任务 B（辅助任务 / OOD 判别）可以复用 typed_head。
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
        self.base = GCTPPStructBackbone(
            in_channels=in_channels,
            graph_hidden_dim=graph_hidden_dim,
            time_hidden_dim=time_hidden_dim,
            K=K,
            max_history_len=max_history_len,
            dropout_prob=dropout_prob,
        )
        self.max_history_len = max_history_len

        num_coarse = len(COARSE_LABELS)
        self.typed_head = nn.Linear(graph_hidden_dim, num_coarse)

    def forward(self, X_snapshots: torch.Tensor, edge_index: torch.Tensor, dt_history: torch.Tensor):
        mu_nodes, log_sigma_nodes, lambda_nodes, H_last = self.base(
            X_snapshots, edge_index, dt_history
        )
        typed_logits = self.typed_head(H_last)  # (N, C)
        return mu_nodes, log_sigma_nodes, lambda_nodes, typed_logits


# ===========================
# 2. 主流程：Profile-Augmented Struct-Typed（无残差）
# ===========================
def run_gc_tpp_struct_typed(
    data_mode: str = "icews0515",
    seed: int = 0,
    protocol: str = "protB",
    out_dir: str = "logs/table_runs",
):
    """
    Profile-Augmented Struct-Typed 主入口：
    - 在节点特征里拼接 train-only 的 coarse profile；
    - 不对 mu/log_sigma 做任何 Typed 残差；
    - 评估 Seen / OOD / per-coarse 的 NLL & RMSE；
    - Test 结束后落盘 metrics.json（Table1/Table2 自动汇总用）。
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_history_len = 64
    T_snap = 20
    train_ratio = 0.7
    val_ratio = 0.15
    mode_tag = "toy" if data_mode == "toy" else data_mode

    # ---------- 1. Load Data ----------
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

    # ---------- 1.1 Profile 注入 ----------
    N = X_snapshots.size(1)
    profile_path = f"./data/node_profile_{data_mode}.npy"
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"Profile file {profile_path} not found. "
            f"Please run tools/build_node_profile.py first."
        )

    node_profile_np = np.load(profile_path)  # (N, C_profile)
    if node_profile_np.shape[0] != N:
        raise ValueError(
            f"node_profile N mismatch: profile N={node_profile_np.shape[0]}, X_snapshots N={N}"
        )

    node_profile = torch.from_numpy(node_profile_np).to(X_snapshots.device).float()  # (N, C_profile)
    C_profile = node_profile.size(1)
    T_snap = X_snapshots.size(0)
    coarse_rep = node_profile.unsqueeze(0).expand(T_snap, -1, -1)  # (T_snap, N, C_profile)

    X_snapshots = torch.cat([X_snapshots, coarse_rep], dim=-1)  # (T_snap, N, F_in + C_profile)
    print(f"[Profile] Injected node_profile: X_snapshots now {X_snapshots.shape}")

    # ---------- 2. Coarse Types（仅用于 per-coarse 分析） ----------
    ds_for_coarse = GC_TPP_Dataset(
        snapshots_dir="./data/snapshots",
        events_dir="./data/events",
        T=T_snap,
        N=X_snapshots.size(1),
        F_in=X_snapshots.size(-1),
        device=device,
        mode=data_mode,
        save_to_disk=True,
    )
    coarse_all = ds_for_coarse.get_event_coarse_types().to(device)

    # ---------- 3. Model ----------
    in_channels = X_snapshots.size(-1)
    model = GCTPPStructTyped(
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

    print(f"[Profile-Struct-Typed] Start Training ({mode_tag})... seed={seed} protocol={protocol}")

    num_epochs = 15
    train_nll_list, val_nll_list = [], []
    train_rmse_list, val_rmse_list = [], []
    train_mae_list, val_mae_list = [], []

    # ---------- 4. Train + Val ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_total_nll = 0.0
        train_se_sum = 0.0
        train_ae_sum = 0.0
        train_count = 0

        for i in range(dt_train.numel() - 1):
            optimizer.zero_grad()
            start_idx = max(0, i - max_history_len + 1)
            dt_history = dt_train[start_idx: i + 1]
            dt_next = dt_train[i + 1]

            mu_nodes, log_sigma_nodes, lambda_nodes, typed_logits = model(
                X_snapshots, edge_index, dt_history
            )

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

        avg_train_nll = train_total_nll / max(train_count, 1)
        train_rmse = math.sqrt(train_se_sum / max(train_count, 1))
        train_mae = train_ae_sum / max(train_count, 1)

        train_nll_list.append(avg_train_nll)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)

        # ----- Val -----
        model.eval()
        val_total_nll = 0.0
        val_se_sum = 0.0
        val_ae_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for i in range(dt_val.numel() - 1):
                start_idx = max(0, i - max_history_len + 1)
                dt_history = dt_val[start_idx: i + 1]
                dt_next = dt_val[i + 1]

                mu_nodes, log_sigma_nodes, lambda_nodes, typed_logits = model(
                    X_snapshots, edge_index, dt_history
                )

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

        avg_val_nll = val_total_nll / max(val_count, 1)
        val_rmse = math.sqrt(val_se_sum / max(val_count, 1))
        val_mae = val_ae_sum / max(val_count, 1)

        val_nll_list.append(avg_val_nll)
        val_rmse_list.append(val_rmse)
        val_mae_list.append(val_mae)

        print(
            f"Epoch {epoch:02d} | Profile-Struct-Typed ({mode_tag}) "
            f"Train NLL = {avg_train_nll:.4f} | Val NLL = {avg_val_nll:.4f} "
            f"| Train RMSE = {train_rmse:.4f} | Val RMSE = {val_rmse:.4f}"
        )

        scheduler.step(avg_val_nll)
        if early_stopper.step(avg_val_nll):
            print(f"[Profile-Struct-Typed] Early stopping at epoch {epoch}. Best Val NLL = {early_stopper.best:.4f}")
            break

    # ---------- 5. Test (Seen / OOD) ----------
    model.eval()

    test_total_nll = 0.0
    test_se_sum = 0.0
    test_ae_sum = 0.0
    test_count = 0
    test_cov90_sum = 0.0

    test_total_nll_seen = 0.0
    test_se_sum_seen = 0.0
    test_ae_sum_seen = 0.0
    test_count_seen = 0
    test_cov90_sum_seen = 0.0

    test_total_nll_ood = 0.0
    test_se_sum_ood = 0.0
    test_ae_sum_ood = 0.0
    test_count_ood = 0
    test_cov90_sum_ood = 0.0

    seen_test = flags["seen_test"] if isinstance(flags, dict) and "seen_test" in flags else None
    ood_test = flags["ood_test"] if isinstance(flags, dict) and "ood_test" in flags else None

    coarse_stats = {}

    def _update_coarse(coarse_id: int, tag: str, nll_val: float, se_val: float, ae_val: float, cov90_val: float):
        key = (int(coarse_id), tag)
        if key not in coarse_stats:
            coarse_stats[key] = {"sum_nll": 0.0, "sum_se": 0.0, "sum_ae": 0.0, "sum_cov90": 0.0, "count": 0}
        s = coarse_stats[key]
        s["sum_nll"] += nll_val
        s["sum_se"] += se_val
        s["sum_ae"] += ae_val
        s["sum_cov90"] += cov90_val
        s["count"] += 1

    # 90% interval 的 z 值（标准正态的 0.95 分位）
    z_90 = 1.6448536269514722

    with torch.no_grad():
        for i in range(dt_test.numel() - 1):
            start_idx = max(0, i - max_history_len + 1)
            dt_history = dt_test[start_idx: i + 1]
            dt_next = dt_test[i + 1]

            mu_nodes, log_sigma_nodes, lambda_nodes, typed_logits = model(
                X_snapshots, edge_index, dt_history
            )

            # NLL
            nll_nodes = lognormal_nll(dt_next, mu_nodes, log_sigma_nodes)
            nll = nll_nodes.mean()

            # RMSE/MAE on logΔt
            log_dt_true = torch.log(torch.clamp(dt_next, min=1e-8))
            log_dt_pred_nodes = mu_nodes
            err_log_nodes = log_dt_pred_nodes - log_dt_true

            se = float(torch.mean(err_log_nodes ** 2))
            ae = float(torch.mean(torch.abs(err_log_nodes)))

            # Coverage@90（按“每个事件：所有节点预测区间里包含真实 dt 的比例”，再对事件平均）
            sigma_nodes = torch.exp(log_sigma_nodes)
            lower_log = mu_nodes - z_90 * sigma_nodes
            upper_log = mu_nodes + z_90 * sigma_nodes
            lower_dt = torch.exp(lower_log)
            upper_dt = torch.exp(upper_log)
            inside = ((dt_next >= lower_dt) & (dt_next <= upper_dt)).float()
            cov90_event = float(inside.mean().item())

            test_se_sum += se
            test_ae_sum += ae
            test_total_nll += float(nll.item())
            test_cov90_sum += cov90_event
            test_count += 1

            ev_idx = i + 1

            coarse_id = None
            if ev_idx < int(coarse_all.numel()):
                coarse_id = int(coarse_all[ev_idx].item())
                _update_coarse(coarse_id, "all", float(nll.item()), se, ae, cov90_event)

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
                test_cov90_sum_seen += cov90_event
                test_count_seen += 1
                if coarse_id is not None:
                    _update_coarse(coarse_id, "seen", float(nll.item()), se, ae, cov90_event)

            if is_ood:
                test_se_sum_ood += se
                test_ae_sum_ood += ae
                test_total_nll_ood += float(nll.item())
                test_cov90_sum_ood += cov90_event
                test_count_ood += 1
                if coarse_id is not None:
                    _update_coarse(coarse_id, "ood", float(nll.item()), se, ae, cov90_event)

    def _safe_mean(total, count):
        return total / count if count > 0 else float("nan")

    avg_test_nll = _safe_mean(test_total_nll, test_count)
    test_rmse = math.sqrt(_safe_mean(test_se_sum, test_count)) if test_count > 0 else float("nan")
    test_mae = _safe_mean(test_ae_sum, test_count)
    test_cov90 = _safe_mean(test_cov90_sum, test_count)

    avg_test_nll_seen = _safe_mean(test_total_nll_seen, test_count_seen)
    test_rmse_seen = math.sqrt(_safe_mean(test_se_sum_seen, test_count_seen)) if test_count_seen > 0 else float("nan")
    test_mae_seen = _safe_mean(test_ae_sum_seen, test_count_seen)
    test_cov90_seen = _safe_mean(test_cov90_sum_seen, test_count_seen)

    avg_test_nll_ood = _safe_mean(test_total_nll_ood, test_count_ood)
    test_rmse_ood = math.sqrt(_safe_mean(test_se_sum_ood, test_count_ood)) if test_count_ood > 0 else float("nan")
    test_mae_ood = _safe_mean(test_ae_sum_ood, test_count_ood)
    test_cov90_ood = _safe_mean(test_cov90_sum_ood, test_count_ood)

    print(f"[Profile-Struct-Typed] Test NLL  (all) = {avg_test_nll:.4f}")
    print(f"[Profile-Struct-Typed] Test RMSE (log Δt, all) = {test_rmse:.4f}")
    print(f"[Profile-Struct-Typed] Test MAE  (log Δt, all) = {test_mae:.4f}")
    print(f"[Profile-Struct-Typed] Test Cov90 (all) = {test_cov90:.4f}")

    print(f"[Profile-Struct-Typed] Test NLL  (Seen) = {avg_test_nll_seen:.4f} | count={test_count_seen}")
    print(f"[Profile-Struct-Typed] Test RMSE (Seen) = {test_rmse_seen:.4f} | count={test_count_seen}")
    print(f"[Profile-Struct-Typed] Test MAE  (Seen) = {test_mae_seen:.4f} | count={test_count_seen}")
    print(f"[Profile-Struct-Typed] Test Cov90 (Seen) = {test_cov90_seen:.4f} | count={test_count_seen}")

    print(f"[Profile-Struct-Typed] Test NLL  (OOD)  = {avg_test_nll_ood:.4f} | count={test_count_ood}")
    print(f"[Profile-Struct-Typed] Test RMSE (OOD)  = {test_rmse_ood:.4f} | count={test_count_ood}")
    print(f"[Profile-Struct-Typed] Test MAE  (OOD)  = {test_mae_ood:.4f} | count={test_count_ood}")
    print(f"[Profile-Struct-Typed] Test Cov90 (OOD) = {test_cov90_ood:.4f} | count={test_count_ood}")

    # ---------- 保存 per-coarse 指标 CSV ----------
    os.makedirs("logs", exist_ok=True)
    coarse_csv_path = f"logs/gc_tpp_struct_typed_profile_{data_mode}.csv"
    with open(coarse_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["coarse_id", "coarse_name", "tag", "count", "nll_mean", "rmse_mean", "mae_mean", "cov90_mean"]
        )
        for (cid, tag), s in sorted(coarse_stats.items()):
            cnt = s["count"]
            if cnt <= 0:
                continue
            nll_mean = s["sum_nll"] / cnt
            rmse_mean = math.sqrt(s["sum_se"] / cnt)
            mae_mean = s["sum_ae"] / cnt
            cov90_mean = s["sum_cov90"] / cnt
            cname = COARSE_LABELS.get(cid, f"coarse_{cid}")
            writer.writerow([cid, cname, tag, cnt, nll_mean, rmse_mean, mae_mean, cov90_mean])

    print(f"[Profile-Struct-Typed] Saved coarse-level metrics to {coarse_csv_path}")

    # ---------- 写 metrics.json（Table1/Table2 汇总用） ----------
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_name = "gc_tpp_struct_typed"
    save_dir = os.path.join(out_dir, protocol, model_name, f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.json")

    # Table2: 只要 OOD 的按 coarse 分组
    by_coarse_type_ood = []
    for (cid, tag), s in sorted(coarse_stats.items()):
        if tag != "ood":
            continue
        cnt = int(s["count"])
        if cnt <= 0:
            continue
        by_coarse_type_ood.append(
            {
                "coarse_id": int(cid),
                "coarse_type": COARSE_LABELS.get(int(cid), f"coarse_{cid}"),
                "count": cnt,
                "time_nll": float(s["sum_nll"] / cnt),
                "logdt_rmse": float(math.sqrt(s["sum_se"] / cnt)),
                "logdt_mae": float(s["sum_ae"] / cnt),
                "cov90": float(s["sum_cov90"] / cnt),
            }
        )

    payload = {
        "meta": {
            "protocol": protocol,
            "data_mode": data_mode,
            "model": model_name,
            "seed": int(seed),
            "timestamp": ts,
        },
        "summary": {
            "test": {
                "all": {
                    "count": int(test_count),
                    "time_nll": float(avg_test_nll),
                    "logdt_rmse": float(test_rmse),
                    "logdt_mae": float(test_mae),
                    "cov90": float(test_cov90),
                },
                "seen": {
                    "count": int(test_count_seen),
                    "time_nll": float(avg_test_nll_seen),
                    "logdt_rmse": float(test_rmse_seen),
                    "logdt_mae": float(test_mae_seen),
                    "cov90": float(test_cov90_seen),
                },
                "ood": {
                    "count": int(test_count_ood),
                    "time_nll": float(avg_test_nll_ood),
                    "logdt_rmse": float(test_rmse_ood),
                    "logdt_mae": float(test_mae_ood),
                    "cov90": float(test_cov90_ood),
                },
            }
        },
        "by_coarse_type_ood": by_coarse_type_ood,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Profile-Struct-Typed] Wrote metrics.json -> {metrics_path}")

    # ---------- 保存 npz ----------
    if data_mode == "toy":
        save_path = "logs/gc_tpp_struct_typed_profile_toy.npz"
    elif data_mode == "icews_real":
        save_path = "logs/gc_tpp_struct_typed_profile_icews_real.npz"
    else:
        save_path = f"logs/gc_tpp_struct_typed_profile_{data_mode}.npz"

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
        test_cov90=test_cov90,
        test_nll_seen=avg_test_nll_seen,
        test_rmse_seen=test_rmse_seen,
        test_mae_seen=test_mae_seen,
        test_cov90_seen=test_cov90_seen,
        test_nll_ood=avg_test_nll_ood,
        test_rmse_ood=test_rmse_ood,
        test_mae_ood=test_mae_ood,
        test_cov90_ood=test_cov90_ood,
    )
    print(f"[Profile-Struct-Typed] Saved curves to {save_path}")

    return {
        "data_mode": data_mode,
        "test_nll": avg_test_nll,
        "test_rmse": test_rmse,
        "test_nll_seen": avg_test_nll_seen,
        "test_rmse_seen": test_rmse_seen,
        "test_nll_ood": avg_test_nll_ood,
        "test_rmse_ood": test_rmse_ood,
        "coarse_csv_path": coarse_csv_path,
        "log_path": save_path,
        "metrics_path": metrics_path,
    }

