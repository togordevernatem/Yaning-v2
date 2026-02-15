import torch
import torch.nn as nn


class GRULogNormalBaseline(nn.Module):
    """
    Event-sequence GRU baseline:
      input: (src, dst, coarse_type, log_dt_prev) sequence
      output: (mu, log_sigma) for next log-dt
    """

    def __init__(
        self,
        num_nodes: int,
        num_coarse_types: int,
        node_emb_dim: int = 32,
        type_emb_dim: int = 16,
        dt_emb_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.type_emb = nn.Embedding(num_coarse_types, type_emb_dim)
        self.dt_mlp = nn.Sequential(
            nn.Linear(1, dt_emb_dim),
            nn.ReLU(),
            nn.Linear(dt_emb_dim, dt_emb_dim),
        )

        in_dim = node_emb_dim * 2 + type_emb_dim + dt_emb_dim
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [mu, log_sigma]
        )

    def forward(self, src_seq, dst_seq, type_seq, log_dt_seq):
        """
        src_seq, dst_seq, type_seq, log_dt_seq: shape (L,)
        Returns: mu, log_sigma (scalars)
        """
        src_e = self.node_emb(src_seq)
        dst_e = self.node_emb(dst_seq)
        typ_e = self.type_emb(type_seq)

        log_dt_seq = log_dt_seq.unsqueeze(-1)  # (L, 1)
        dt_e = self.dt_mlp(log_dt_seq)         # (L, dt_emb_dim)

        x = torch.cat([src_e, dst_e, typ_e, dt_e], dim=-1)  # (L, in_dim)
        x = x.unsqueeze(0)  # (1, L, in_dim)

        _, h = self.gru(x)  # h: (num_layers, 1, hidden_dim)
        h_last = h[-1, 0]   # (hidden_dim,)

        out = self.head(h_last)  # (2,)
        mu = out[0]
        log_sigma = out[1].clamp(min=-6.0, max=3.0)  # 稳定训练
        return mu, log_sigma
