import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.data import Data

# 1. 设备设置：现在是 CPU，将来上 GPU 也只改这里
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 2. 构造一个玩具“动态图序列”
# 假设有 N 个节点，每个时间步都有 X_t (N, F)，边结构固定不变
T = 20          # 时间步数
N = 10          # 节点数
F_in = 3        # 输入特征维度
F_hidden = 8    # GConvGRU 隐层维度
F_out = 2       # 输出特征维度（比如 2 类分类，也可以当作回归维度）

# 用一个非常简单的有向链式图：0->1->2->...->N-1
edge_index = torch.tensor(
    [
        [i for i in range(N - 1)],      # 源节点
        [i + 1 for i in range(N - 1)]   # 目标节点
    ],
    dtype=torch.long
)

edge_index = edge_index.to(device)

# 生成一个“时间序列”的节点特征：这里先用随机数占位
X_list = []
Y_list = []
for t in range(T):
    x_t = torch.randn(N, F_in)          # 当前时刻特征
    y_t = torch.randint(0, F_out, (N,)) # 每个点一个分类标签，取值 0 或 1（如果 F_out=2）
    X_list.append(x_t)
    Y_list.append(y_t)

# 搬到 device
X_list = [x.to(device) for x in X_list]
Y_list = [y.to(device) for y in Y_list]

# 3. 定义一个“GConvGRU + 线性头”的简单模型
class GConvGRUClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=2):
        super().__init__()
        # 官方 GConvGRU：内部自动做图卷积 + GRU 门控
        self.gconv_gru = GConvGRU(in_channels=in_channels,
                                  out_channels=hidden_channels,
                                  K=K)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, X_seq, edge_index):
        """
        X_seq: list[Tensor(N, F)]，这里用 list 长度为 T，每个是 (N, F)
        edge_index: (2, E)
        """
        h = None
        # 逐时间步送入 GConvGRU
        for x_t in X_seq:
            h = self.gconv_gru(x_t, edge_index, H=h)

        # 用最后一个时间步的 hidden 做分类
        out = self.linear(h)  # (N, out_channels)
        return out

model = GConvGRUClassifier(
    in_channels=F_in,
    hidden_channels=F_hidden,
    out_channels=F_out,
    K=2,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 4. 一个非常简单的训练循环，只为测试环境 & API
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # 这里用前 T-1 步作为输入，最后一步作为监督标签
    logits = model(X_list[:-1], edge_index)  # (N, F_out)
    y_target = Y_list[-1]                    # (N,)

    loss = criterion(logits, y_target)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch:02d} | Loss = {loss.item():.4f}")

print("[INFO] Training finished. GConvGRU baseline toy is OK.")

