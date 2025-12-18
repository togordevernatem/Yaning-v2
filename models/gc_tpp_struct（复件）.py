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
        max_history_len: int = 256,
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
        # 1) 图编码：拿到最后一张 snapshot 的节点表示 H_last
        g_t, H_all = self.graph_encoder(X_snapshots, edge_index, return_node_repr=True)
        H_last = H_all[-1]  # (N, hidden_dim)

        # 2) 时间编码
        h_t = self.time_encoder(dt_history)  # (time_hidden_dim,)
        h_rep = h_t.unsqueeze(0).expand(H_last.size(0), -1)  # (N, time_hidden_dim)

        # 3) 拼接得到每个节点的特征
        z_nodes = torch.cat([H_last, h_rep], dim=-1)  # (N, fusion_dim)

        # 4) 节点级 [mu_i, log_sigma_i]
        out = self.node_mlp(z_nodes)  # (N, 2)
        mu_nodes = out[:, 0]          # (N,)
        log_sigma_nodes = out[:, 1]   # (N,)

        # 5) 节点 λ_i(t)，用于 debug 观察
        lambda_nodes = torch.exp(mu_nodes + 0.5 * torch.exp(2 * log_sigma_nodes))

        return mu_nodes, log_sigma_nodes, lambda_nodes


# ===========================
# 2. 从 Dataset 构建 Train/Val/Test
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

    (X_list,
     edge_index,
     event_times_train,
     dt_train,
     event_times_val,
     dt_val,
     event_times_test,
     dt_test) = ds.get_train_val_test_split(train_ratio=train_ratio, val_ratio=val_ratio)

    X_tensor = torch.stack(X_list, dim=0).to(device)  # (T_snap, N, F_in)
    edge_index = edge_index.to(device)

    return X_tensor, edge_index, event_times_train, dt_train, event_times_val, dt_val, event_times_test, dt_test


# ===========================
# 3. 训练 & 验证 & 测试 主流程（Struct）
# ===========================
def run_gc_tpp_struct(data_mode: str = "toy"):
    """
    Struct 版本的训练流程：
    - 损失仍然是节点级 LogNormal NLL（对节点求平均）
    - RMSE/MAE 改为在 log(Δt) 空间计算：比较 mu_i 和 log(dt_next)
      （这里把所有节点的误差做平均）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_history_len = 256
    T_snap = 20

    mode_tag = "toy" if data_mode == "toy" else data_mode

    # 1) 取数据
    (X_snapshots,
     edge_index,
     event_times_train,
     dt_train,
     event_times_val,
     dt_val,
     event_times_test,
     dt_test) = build_events_from_dataset(device=device, T_snap=T_snap, mode=data_mode)

    # 2) 初始化模型
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
    early_stopper = EarlyStopping(patience=7, min_delta=1e-3)

    num_epochs = 50

    train_nll_list = []
    val_nll_list = []
    train_rmse_list = []
    val_rmse_list = []
    train_mae_list = []
    val_mae_list = []

    print(f"[INFO] Using device: {device}")
    print("[INFO] Total events = {}, train_events = {}, val_events = {}, test_events = {}".format(
        dt_train.numel(),
        dt_train.numel() - 1 - (dt_val.numel() - 1) - (dt_test.numel() - 1),
        dt_val.numel() - 1,
        dt_test.numel() - 1,
    ))

    # 3) 训练 + 验证
    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
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

            mu_nodes, log_sigma_nodes, lambda_nodes = model(X_snapshots, edge_index, dt_history)

            # 节点级 NLL，取平均
            nll_nodes = lognormal_nll(dt_next, mu_nodes, log_sigma_nodes)  # (N,)
            nll = nll_nodes.mean()

            # log(Δt) 空间误差：对所有节点平均
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
            debug_lambda = lambda_nodes  # 用最后一次的 λ_i(t) 打印示例

        avg_train_nll = train_total_nll / max(train_count, 1)
        train_rmse = math.sqrt(train_se_sum / max(train_count, 1))
        train_mae = train_ae_sum / max(train_count, 1)

        train_nll_list.append(avg_train_nll)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)

        # ---- Val ----
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

                mu_nodes, log_sigma_nodes, lambda_nodes = model(X_snapshots, edge_index, dt_history)

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

        # 输出训练和验证结果
        print(
            f"Epoch {epoch:02d} | GC-TPP-Struct ({mode_tag}) "
            f"Train NLL = {avg_train_nll:.4f} | Val NLL = {avg_val_nll:.4f} "
            f"| Train RMSE (log Δt) = {train_rmse:.4f} | Val RMSE (log Δt) = {val_rmse:.4f} "
            f"| Train MAE (log Δt) = {train_mae:.4f} | Val MAE (log Δt) = {val_mae:.4f}"
        )

        # 打印一个节点 λ_i(t) 的示例（前 5 个）
        if debug_lambda_val is not None:
            lam_np = debug_lambda_val.detach().cpu().numpy()
            print(f"    [DEBUG] 示例节点 λ_i(t) (前 5 个) = {lam_np[:5]}")

        # 学习率调度 & 早停
        scheduler.step(avg_val_nll)
        if early_stopper.step(avg_val_nll):
            print(f"[INFO] Early stopping triggered at epoch {epoch}. Best Val NLL = {early_stopper.best:.4f}")
            break

    # 4) 测试集评估
    model.eval()
    test_total_nll = 0.0
    test_se_sum = 0.0
    test_ae_sum = 0.0
    test_count = 0

    with torch.no_grad():
        for i in range(dt_test.numel() - 1):
            start_idx = max(0, i - max_history_len + 1)
            dt_history = dt_test[start_idx: i + 1]
            dt_next = dt_test[i + 1]

            mu_nodes, log_sigma_nodes, lambda_nodes = model(X_snapshots, edge_index, dt_history)

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

    avg_test_nll = test_total_nll / max(test_count, 1)
    test_rmse = math.sqrt(test_se_sum / max(test_count, 1))
    test_mae = test_ae_sum / max(test_count, 1)

    print(f"[INFO] Test NLL  (Struct {mode_tag})         = {avg_test_nll:.4f}")
    print(f"[INFO] Test RMSE (Struct {mode_tag}, log Δt) = {test_rmse:.4f}")
    print(f"[INFO] Test MAE  (Struct {mode_tag}, log Δt) = {test_mae:.4f}")

    # 5) 保存曲线
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
    )
    print(f"[INFO] Saved Struct {mode_tag} curves to {save_path}")

