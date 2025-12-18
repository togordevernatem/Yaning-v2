import os
import math
import numpy as np
import torch
import torch.nn as nn

from torch_geometric_temporal.nn.recurrent import GConvGRU
from data.dataset_toy import GC_TPP_Dataset


# ===========================
# 0. EarlyStopping 小工具
# ===========================
class EarlyStopping:
    """
    基于验证集 NLL 的早停，不依赖 RMSE/MAE，所以 RMSE/MAE 即使换成 log 空间也不影响训练逻辑。
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, current: float) -> bool:
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience


# ===========================
# 1. LogNormal NLL（用来评估 Δt）
# ===========================
def lognormal_nll(dt, mu, log_sigma, eps: float = 1e-8) -> torch.Tensor:
    """
    dt:       标量或张量 (实数 > 0)
    mu:       对应 log(dt) 的均值
    log_sigma: 对应 log(dt) 的 log 标准差
    """
    dt = torch.clamp(dt, min=eps)
    sigma = torch.exp(log_sigma)
    log_dt = torch.log(dt)
    nll = 0.5 * ((log_dt - mu) / sigma) ** 2 + log_dt + log_sigma
    return nll


def lognormal_mean(mu, log_sigma):
    """
    E[dt] 的解析表达式（如果以后想在原始 Δt 空间做估计仍然可以用）。
    """
    sigma = torch.exp(log_sigma)
    return torch.exp(mu + 0.5 * sigma ** 2)


# ===========================
# 2. 图编码器：GConvGRU
# ===========================
class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = None, hidden_channels: int = None, K: int = 3):
        super().__init__()
        if hidden_dim is None and hidden_channels is None:
            raise ValueError("GraphEncoder 需要传入 hidden_dim 或 hidden_channels 之一")
        if hidden_dim is None:
            hidden_dim = hidden_channels

        self.hidden_dim = hidden_dim
        self.gconvgru = GConvGRU(in_channels=in_channels, out_channels=self.hidden_dim, K=K)

    def forward(self, X_seq: torch.Tensor, edge_index: torch.Tensor, return_node_repr: bool = False):
        """
        X_seq: (T_snap, N, F_in)
        edge_index: (2, E)
        return_node_repr:
          - False: 只返回图级表示 g_t
          - True:  返回 (g_t, H_all)，其中 H_all 是长度为 T_snap 的列表，每个元素形状 (N, hidden_dim)
        """
        H = None
        H_all = []
        for t in range(X_seq.size(0)):
            x_t = X_seq[t]
            H = self.gconvgru(x_t, edge_index, None, H, None)  # (N, hidden_dim)
            H_all.append(H)

        # 图级池化：对最后一个时间步的节点表示取平均
        g_t = H.mean(dim=0)  # (hidden_dim,)

        if return_node_repr:
            return g_t, H_all
        else:
            return g_t


# ===========================
# 3. 时间编码器：GRU on Δt
# ===========================
class TimeEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=1, hidden_size=self.hidden_dim, batch_first=True)

    def forward(self, dt_history: torch.Tensor) -> torch.Tensor:
        # dt_history: (L_hist,)
        x = dt_history.view(1, -1, 1)  # (1, L_hist, 1)
        _, h = self.gru(x)            # h: (1, hidden_dim)
        h = h.view(-1)                # (hidden_dim,)
        return h


# ===========================
# 4. GC-TPP 主模型（Core + 正则/Dropout）
# ===========================
class GCTPPContinuous(nn.Module):
    def __init__(self, in_channels: int, graph_hidden_dim: int = 32, time_hidden_dim: int = 32, K: int = 3,
                 max_history_len: int = 256, dropout_prob: float = 0.2):
        super().__init__()
        self.max_history_len = max_history_len
        self.graph_encoder = GraphEncoder(in_channels=in_channels, hidden_dim=graph_hidden_dim, K=K)
        self.time_encoder = TimeEncoder(hidden_dim=time_hidden_dim)

        fusion_dim = graph_hidden_dim + time_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_dim, 2),  # 输出 [mu, log_sigma]
        )

    def forward(self, X_snapshots: torch.Tensor, edge_index: torch.Tensor, dt_history: torch.Tensor):
        """
        X_snapshots: (T_snap, N, F_in)
        edge_index:  (2, E)
        dt_history:  (L_hist,)
        """
        g_t = self.graph_encoder(X_snapshots, edge_index)  # (graph_hidden_dim,)
        h_t = self.time_encoder(dt_history)                # (time_hidden_dim,)

        z = torch.cat([g_t, h_t], dim=-1).view(1, -1)
        out = self.mlp(z).view(-1)

        mu = out[0]
        log_sigma = out[1]

        # λ(t) = E[1/Δt] 只是一个示意，这里保留之前定义
        lambda_t = torch.exp(mu + 0.5 * torch.exp(2 * log_sigma))

        return mu, log_sigma, lambda_t


# ===========================
# 5. 从 Dataset 构建 Train/Val/Test
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
# 6. 训练 & 验证 & 测试 主流程（含 RMSE/MAE + EarlyStopping）
# ===========================
def run_gc_tpp_continuous(data_mode: str = "toy"):
    """
    注意：这里的 RMSE/MAE 已经改为在 log(Δt) 空间计算，
    即比较 mu（预测的 log Δt 均值）和 log(dt_next)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_history_len = 256
    T_snap = 20

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
    model = GCTPPContinuous(
        in_channels=in_channels,
        graph_hidden_dim=32,
        time_hidden_dim=32,
        K=3,
        max_history_len=max_history_len,
        dropout_prob=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # L2 正则
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stopper = EarlyStopping(patience=7, min_delta=1e-3)

    num_epochs = 50

    train_nll_list = []
    val_nll_list = []
    train_rmse_list = []
    val_rmse_list = []
    train_mae_list = []
    val_mae_list = []

    # 3) 训练 + 验证
    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
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

            mu, log_sigma, lambda_t = model(X_snapshots, edge_index, dt_history)
            nll = lognormal_nll(dt_next, mu, log_sigma)

            # ---- 在 log(Δt) 空间计算误差，避免溢出 ----
            log_dt_true = torch.log(torch.clamp(dt_next, min=1e-8))
            log_dt_pred = mu.detach()  # mu 就是 log-normal 的均值，对应 log Δt
            err_log = log_dt_pred - log_dt_true

            train_se_sum += float(err_log ** 2)
            train_ae_sum += float(torch.abs(err_log))

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

        # ---- Val ----
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

                mu, log_sigma, lambda_t = model(X_snapshots, edge_index, dt_history)
                nll = lognormal_nll(dt_next, mu, log_sigma)

                # log(Δt) 空间误差
                log_dt_true = torch.log(torch.clamp(dt_next, min=1e-8))
                log_dt_pred = mu
                err_log = log_dt_pred - log_dt_true

                val_se_sum += float(err_log ** 2)
                val_ae_sum += float(torch.abs(err_log))

                val_total_nll += float(nll.item())
                val_count += 1

        avg_val_nll = val_total_nll / max(val_count, 1)
        val_rmse = math.sqrt(val_se_sum / max(val_count, 1))
        val_mae = val_ae_sum / max(val_count, 1)

        val_nll_list.append(avg_val_nll)
        val_rmse_list.append(val_rmse)
        val_mae_list.append(val_mae)

        # 输出训练和验证结果
        print(
            f"Epoch {epoch:02d} | GC-TPP continuous "
            f"Train NLL = {avg_train_nll:.4f} | Val NLL = {avg_val_nll:.4f} "
            f"| Train RMSE (log Δt) = {train_rmse:.4f} | Val RMSE (log Δt) = {val_rmse:.4f} "
            f"| Train MAE (log Δt) = {train_mae:.4f} | Val MAE (log Δt) = {val_mae:.4f}"
        )

        # 学习率调度
        scheduler.step(avg_val_nll)

        # Early stopping
        if early_stopper.step(avg_val_nll):
            print(f"[INFO] Early stopping triggered at epoch {epoch}. Best Val NLL = {early_stopper.best:.4f}")
            break

    # 4) 测试集评估（用最后一轮参数）
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

            mu, log_sigma, lambda_t = model(X_snapshots, edge_index, dt_history)
            nll = lognormal_nll(dt_next, mu, log_sigma)

            # log(Δt) 空间误差
            log_dt_true = torch.log(torch.clamp(dt_next, min=1e-8))
            log_dt_pred = mu
            err_log = log_dt_pred - log_dt_true

            test_se_sum += float(err_log ** 2)
            test_ae_sum += float(torch.abs(err_log))

            test_total_nll += float(nll.item())
            test_count += 1

    avg_test_nll = test_total_nll / max(test_count, 1)
    test_rmse = math.sqrt(test_se_sum / max(test_count, 1))
    test_mae = test_ae_sum / max(test_count, 1)

    print(f"[INFO] Test NLL  (on held-out events) = {avg_test_nll:.4f}")
    print(f"[INFO] Test RMSE (log Δt prediction)  = {test_rmse:.4f}")
    print(f"[INFO] Test MAE  (log Δt prediction)  = {test_mae:.4f}")

    # 5) 保存曲线
    os.makedirs("logs", exist_ok=True)
    if data_mode == "toy":
        save_path = "logs/gc_tpp_core_toy.npz"
    elif data_mode == "icews_real":
        save_path = "logs/gc_tpp_core_icews_real.npz"
    else:
        save_path = f"logs/gc_tpp_core_{data_mode}.npz"

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
    print(f"[INFO] Saved Core curves to {save_path}")






