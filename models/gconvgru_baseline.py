import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU


class GConvGRU_Baseline(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=8):
        super().__init__()
        self.encoder = GConvGRU(in_channels=in_channels,
                                out_channels=hidden_channels,
                                K=2)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, X_list, edge_index):
        H = None
        for x_t in X_list:
            H = self.encoder(x_t, edge_index, H=H)
        g_t = H.mean(dim=0)           # 图级表示
        logit = self.linear(g_t)      # 标量
        return logit.squeeze(-1)


def run_gconvgru_baseline(
    T: int = 20,
    N: int = 10,
    F_in: int = 3,
    epochs: int = 20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 构造简单链式图
    edge_index = torch.tensor(
        [
            [i for i in range(N - 1)],
            [i + 1 for i in range(N - 1)],
        ],
        dtype=torch.long,
        device=device,
    )

    # 构造 T 个快照特征
    X_list = []
    for _ in range(T):
        x_t = torch.randn(N, F_in, device=device)
        X_list.append(x_t)

    # 目标：每个时间序列对应一个标签（这里随便造一个二分类标签）
    y = torch.randint(low=0, high=2, size=(1,), dtype=torch.float32, device=device)

    model = GConvGRU_Baseline(in_channels=F_in, hidden_channels=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logit = model(X_list, edge_index)
        loss = criterion(logit.unsqueeze(0), y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss = {loss.item():.4f}")

    print("[INFO] Training finished. GConvGRU baseline toy is OK.")
