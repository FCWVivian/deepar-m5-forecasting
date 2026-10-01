"""Global Transformer encoder forecaster (the capstone's model, fixed and trained across all series).

The sequence is 112 past days followed by 28 future days. Past tokens carry scaled sales; future tokens
carry only what is known in advance (calendar, prices, SNAP) plus a flag. Self-attention runs over the
time axis (batch_first=True), and the 28 future tokens are decoded directly into negative-binomial
parameters, using the same μ parameterization as m5/deepar_nb.py so the two models differ only in
architecture.

Capstone bugs this fixes:
  * nn.TransformerEncoderLayer defaulted to batch_first=False while inputs were (batch, time, features),
    so attention mixed the 10 windows in a batch instead of the 196 days of a window;
  * the forecast was decoded from x[:, -196, :], i.e. the oldest day of the window only;
  * the loss covered all 224 outputs (196 reconstructed inputs + 28 forecasts).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerNB(nn.Module):
    def __init__(self, n_items, n_depts, n_stores, n_real=5, context=112, horizon=28,
                 d_model=64, nhead=4, layers=3, dropout=0.1):
        super().__init__()
        self.context, self.horizon = context, horizon
        self.item = nn.Embedding(n_items, 16)
        self.dept = nn.Embedding(n_depts, 4)
        self.store = nn.Embedding(n_stores, 4)
        self.wday = nn.Embedding(7, 3)
        self.month = nn.Embedding(12, 3)
        n_in = 1 + 1 + 1 + n_real + 3 + 3 + 16 + 4 + 4   # sales, is_future, log level, covariates, embeddings
        self.proj = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.randn(context + horizon, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, 3)

    def forward(self, sales_scaled, level, real, wday, month, static):
        """sales_scaled: (B, context) past sales / (level + 1); real/wday/month cover context + horizon."""
        B, T = real.shape[:2]
        sales = torch.cat([sales_scaled, sales_scaled.new_zeros(B, self.horizon)], 1).unsqueeze(-1)
        is_future = torch.cat([real.new_zeros(B, self.context), real.new_ones(B, self.horizon)], 1).unsqueeze(-1)
        s = torch.cat([self.item(static[:, 0]), self.dept(static[:, 1]), self.store(static[:, 2])], -1)
        x = torch.cat([sales, is_future, torch.log1p(level).expand(-1, T).unsqueeze(-1), real,
                       self.wday(wday), self.month(month), s.unsqueeze(1).expand(-1, T, -1)], -1)
        h = self.encoder(self.proj(x) + self.pos)[:, -self.horizon:]
        out = self.head(h)
        mu = F.softplus(out[..., 0] + 0.5413) * level + F.softplus(out[..., 2] - 3)
        alpha = F.softplus(out[..., 1]) / (level + 1).sqrt() + 1e-4
        return mu, alpha
