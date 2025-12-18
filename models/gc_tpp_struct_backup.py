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
# 1. LogNormal NLL（和 Core 一样）
# ===========================
def lognormal_nll(dt, mu, log_sigma, eps: float = 1e-8) -> torch.Tensor:
    dt = torch.clamp(dt, min=eps)
    sigma = torch.exp(log_sigma)
    log_dt = torch.log(dt)
    nll = 0.5 * ((log_dt - mu) / sigma) ** 2 + log_dt + log_sigma
    return nll


def lognormal_mean(mu, log_sigma):
    sigma = torch.exp(log_sigma)
    return torch.exp(mu + 0.5 * sigma ** 2)


# ===========================
# 2. 图编码器：GConvGRU（支持返回 H_all）
# ===========================
class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = None,
                 hidden_channels: int = None, K: int = 3):
        super().__init__()
        if hidden_dim is None and hidden_channels is None:
            raise ValueError("GraphEncoder 需要传入 hidden_dim 或 hidden_channels 之一")
        if hidden_dim is None:
            hidden_dim = hidden_channels

        self.hidden_dim = hidden_dim
        self.gconvgru = GConvGRU(in_channels=in_channels,
                                 out_channels=self.hidden_dim,
                                 K=K)

    def forward(self, X_seq: torch.Tensor, edge_index: torch.Tensor,
                return_node_repr: bool = False):
        """
        X_seq: (T, N, F_in)
        edge_index: (2, E)

        return_node_repr:
          False → 只返回图级表示 g_t
          True  → 返回 (g_t, H_all)，其中 H_all 是每个时间步的节点隐状态列表
        """
        H = None
        H_all = []
        for t in range(X_seq.size(0)):
            x_t = X_seq[t]
            H = self.gconvgru(x_t, edge_index, None, H, None)
            H_all.append(H)        # 每个时间步的 (N, hidden_dim)

        g_t = H.mean(dim=0)       # 图级池化：平均所有节点 (hidden_dim,)

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
        self.gru = nn.GRU(input_size=1,
                          hidden_size=self.hidden_dim,
                          batch_first=True)

    def forward(self, dt_history: torch.Tensor) -> torch.Tensor:
        x = dt_history.view(1, -1, 1)   # (1, L, 1)
        _, h = self.gru(x)              # h: (1, hidden_dim)
        h = h.view(-1)                  # (hidden_dim,)
        return h


# ===========================
# 4. GC-TPP-Struct 主模型
#    - 保留 Core 的全局 Δt 头（mu, log_sigma）
#    - 新增节点强度 λ_i(t) 头（只做一个 demo）
# ===========================
class GCTPPStruct(nn.Module):
    def __init__(self, in_channels: int,
                 graph_hidden_dim: int = 32,
                 time_hidden_dim: int = 32,
                 K: int = 3,
                 max_history_len: int = 256,
                 dropout_prob: float = 0.2):
        super().__init__()

        self.max_history_len = max_history_len
        self.graph_encoder = GraphEncoder(in_channels=in_channels,
                                          hidden_dim=graph_hidden_dim,
                                          K=K)
        self.time_encoder = TimeEncoder(hidden_dim=time_hidden_dim)

        fusion_dim = graph_hidden_dim + time_hidden_dim

        # 全局 Δt 预测头（和 Core 一样）
        self.mlp_global = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_dim, 2),  # [mu, log_sigma]
        )

        # 节点 λ_i(t) 头：对每个节点最后一帧 H_i 做一个 Softplus>0 的强度
        self.node_head = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, 1),
            nn.Softplus(),  # 保证 λ_i(t) > 0
        )

    def forward(self,
                X_snapshots: torch.Tensor,
                edge_index: torch.Tensor,
                dt_history: torch.Tensor):
        """
        返回：
          mu, log_sigma : 全局 Δt LogNormal 参数
          lambda_nodes  : 节点级 λ_i(t)，形状 (N,)
        """
        # 图编码：拿到图级 g_t 以及每个时间步 H_all
        g_t, H_all = self.graph_encoder(X_snapshots,
                                        edge_index,
                                        return_node_repr=True)

        # 这里只是 demo：取“最后一张快照”的节点隐状态 H_last
        H_last = H_all[-1]                        # (N, hidden_dim)

        # 节点 λ_i(t)
        lambda_nodes = self.node_head(H_last).view(-1)  # (N,)

        # 时间编码
        h_t = self.time_encoder(dt_history)

        # 全局 Δt 预测
        z = torch.cat([g_t, h_t], dim=-1).view(1, -1)
        out = self.mlp_global(z).view(-1)

        mu = out[0]
        log_sigma = out[1]

        return mu, log_sigma, lambda_nodes


# ===========================
# 5. 从 Dataset 构建 Train/Val/Test（toy / icews_real 通用）
# ===========================
def build_events_from_dataset(device: torch.device,
                              T_snap: int = 20,
                              mode: str = "toy",
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15):
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

    X_list, edge_index, event_times_train, dt_train, \
        event_times_val, dt_val, \
        event_times_test, dt_test = ds.get_train_val_test_split(
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

    X_tensor = torch.stack(X_list, dim=0).to(device)
    edge_index = edge_index.to(device)

    return (
        X_tensor,
        edge_index,
        event_times_train,
        dt_train,
        event_times_val,
        dt_val,
        event_times_test,
        dt_test,
    )


# ===========================
# 6. 训练 & 验证 & 测试（含节点 λ_i(t) debug + 曲线保存）
# ===========================
def run_gc_tpp_struct(data_mode: str = "toy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_history_len = 256
    T_snap = 20

    print("============================================================")
    print(f"[INFO] Selected model: gc_tpp_struct")
    print(f"[INFO] Data mode (for gc_tpp_struct): {data_mode}")
    print("============================================================")
    print(f"[INFO] Using device: {device}")

    # 1) 取数据（toy / icews_real）
    X_snapshots, edge_index, \
        event_times_train, dt_train, \
        event_times_val, dt_val, \
        event_times_test, dt_test = build_events_from_dataset(
            device=device,
            T_snap=T_snap,
            mode=data_mode
        )

    print(f"[INFO] Total events = {dt_train.numel() + dt_val.numel() + dt_test.numel()}, "
          f"train_events = {dt_train.numel()}, "
          f"val_events = {dt_val.numel()}, "
          f"test_events = {dt_test.numel()}")

    # 2) 初始化模型
    in_channels = X_snapshots.size(-1)
    model = GCTPPStruct(
        in_channels=in_channels,
        graph_hidden_dim=32,
        time_hidden_dim=32,
        K=3,
        max_history_len=max_history_len,
        dropout_prob=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )
    early_stopper = EarlyStopping(patience=7, min_delta=1e-3)

    num_epochs = 30

    # 曲线缓存
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

        sample_lambda = None  # 用来打印示例节点 λ_i(t)

        for i in range(dt_train.numel() - 1):
            optimizer.zero_grad()

            start_idx = max(0, i - max_history_len + 1)
            dt_history = dt_train[start_idx: i + 1]
            dt_next = dt_train[i + 1]

            mu, log_sigma, lambda_nodes = model(X_snapshots, edge_index, dt_history)

            # 记录一个示例 λ_i(t)（最后一次迭代的）
            sample_lambda = lambda_nodes.detach().cpu().numpy()

            nll = lognormal_nll(dt_next, mu, log_sigma)
            dt_pred = lognormal_mean(mu, log_sigma).detach()

            err = dt_pred - dt_next
            train_se_sum += float(err ** 2)
            train_ae_sum += float(torch.abs(err))

            nll.backward()
            optimizer.step()

            train_total_nll += float(nll.item())
            train_count += 1

        avg_train_nll = train_total_nll / max(train_count, 1)
        train_rmse = math.sqrt(train_se_sum / max(train_count, 1))
        train_mae = train_ae_sum / max(train_count, 1)

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

                mu, log_sigma, lambda_nodes = model(X_snapshots, edge_index, dt_history)
                nll = lognormal_nll(dt_next, mu, log_sigma)

                dt_pred = lognormal_mean(mu, log_sigma)

                err = dt_pred - dt_next
                val_se_sum += float(err ** 2)
                val_ae_sum += float(torch.abs(err))

                val_total_nll += float(nll.item())
                val_count += 1

        avg_val_nll = val_total_nll / max(val_count, 1)
        val_rmse = math.sqrt(val_se_sum / max(val_count, 1))
        val_mae = val_ae_sum / max(val_count, 1)

        # 曲线记录
        train_nll_list.append(avg_train_nll)
        val_nll_list.append(avg_val_nll)
        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)
        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)

        # 日志输出
        print(
            f"Epoch {epoch:02d} | GC-TPP-Struct ({data_mode}) "
            f"Train NLL = {avg_train_nll:.4f} | Val NLL = {avg_val_nll:.4f} "
            f"| Train RMSE = {train_rmse:.4f} | Val RMSE = {val_rmse:.4f} "
            f"| Train MAE = {train_mae:.4f} | Val MAE = {val_mae:.4f}"
        )

        if sample_lambda is not None:
            print(f"    [DEBUG] 示例节点 λ_i(t) (前 5 个) = {sample_lambda[:5]}")

        # 学习率调度 + early stopping
        scheduler.step(avg_val_nll)
        if early_stopper.step(avg_val_nll):
            print(f"[INFO] Early stopping triggered at epoch {epoch}. "
                  f"Best Val NLL = {early_stopper.best:.4f}")
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

            mu, log_sigma, lambda_nodes = model(X_snapshots, edge_index, dt_history)
            nll = lognormal_nll(dt_next, mu, log_sigma)

            dt_pred = lognormal_mean(mu, log_sigma)

            err = dt_pred - dt_next
            test_se_sum += float(err ** 2)
            test_ae_sum += float(torch.abs(err))

            test_total_nll += float(nll.item())
            test_count += 1

    avg_test_nll = test_total_nll / max(test_count, 1)
    test_rmse = math.sqrt(test_se_sum / max(test_count, 1))
    test_mae = test_ae_sum / max(test_count, 1)

    print(f"[INFO] Test NLL  (Struct {data_mode}) = {avg_test_nll:.4f}")
    print(f"[INFO] Test RMSE (Struct {data_mode}) = {test_rmse:.4f}")
    print(f"[INFO] Test MAE  (Struct {data_mode}) = {test_mae:.4f}")

    # 5) 保存曲线
    os.makedirs("logs", exist_ok=True)

    suffix = data_mode  # toy → gc_tpp_struct_toy.npz; icews_real → gc_tpp_struct_icews_real.npz
    save_path = os.path.join("logs", f"gc_tpp_struct_{suffix}.npz")

    np.savez(
        save_path,
        train_nll=np.array(train_nll_list, dtype=np.float64),
        val_nll=np.array(val_nll_list, dtype=np.float64),
        train_rmse=np.array(train_rmse_list, dtype=np.float64),
        val_rmse=np.array(val_rmse_list, dtype=np.float64),
        train_mae=np.array(train_mae_list, dtype=np.float64),
        val_mae=np.array(val_mae_list, dtype=np.float64),
        test_nll=float(avg_test_nll),
        test_rmse=float(test_rmse),
        test_mae=float(test_mae),
        data_mode=data_mode,
    )

    print(f"[INFO] Saved Struct {data_mode} curves to {save_path}")

