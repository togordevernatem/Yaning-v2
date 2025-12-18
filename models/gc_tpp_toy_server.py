import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

T = 20
N = 10
F_in = 3
F_hidden = 8

edge_index = torch.tensor(
    [
        [i for i in range(N - 1)],
        [i + 1 for i in range(N - 1)]
    ],
    dtype=torch.long
).to(device)

X_list = []
for t in range(T):
    x_t = torch.randn(N, F_in)
    X_list.append(x_t.to(device))

y_event = torch.randint(low=0, high=2, size=(T,), dtype=torch.float32).to(device)

class GC_TPP_Toy(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = GConvGRU(in_channels=in_channels, out_channels=hidden_channels, K=2)
        self.event_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, X_prefix, edge_index):
        H = None
        for x_t in X_prefix:
            H = self.encoder(x_t, edge_index, H=H)
        g_t = H.mean(dim=0)
        logit_t = self.event_head(g_t)
        return logit_t.squeeze(-1)

model = GC_TPP_Toy(in_channels=F_in, hidden_channels=F_hidden).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    count = 0
    for t in range(1, T):
        X_prefix = X_list[:t]
        target = y_event[t]
        logit_t = model(X_prefix, edge_index)
        loss_t = criterion(logit_t.unsqueeze(0), target.unsqueeze(0))
        loss_sum += loss_t
        count += 1
    loss = loss_sum / count
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch:02d} | GC-TPP-style event loss = {loss.item():.4f}")

print("[INFO] Training finished. GC-TPP + GConvGRU toy is OK.")
