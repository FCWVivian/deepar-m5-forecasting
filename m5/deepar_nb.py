"""Global DeepAR (Salinas et al., 2017) with a negative-binomial likelihood.

One autoregressive LSTM is trained across all 30,490 series. At each step it sees lagged (scaled)
sales from 1, 7, 14 and 28 days earlier, calendar/price covariates and item/store embeddings, and outputs the
mean μ and shape α of a negative binomial, which fits zero-heavy count data far better than MSE.
Inputs are divided by (1 + context mean) as in the paper; μ is a learned multiple of the context
mean plus a small additive term (see `forward`).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import LAST_DAY, load_calendar, load_prices, sales_matrix

LAGS = [1, 7, 14, 28]    # lagged sales fed as inputs (as in the DeepAR paper / GluonTS)


class Arrays:
    """Everything the model needs, as dense arrays on the (series, day) grid."""

    def __init__(self, sales):
        self.y = sales_matrix(sales)                                  # (N, D)
        n, D = self.y.shape
        cal = load_calendar().set_index("d").loc[1:D + 28]            # allow lookup past LAST_DAY
        prices = load_prices().pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
        prices = prices.reindex(pd.MultiIndex.from_frame(sales[["store_id", "item_id"]]))
        price = prices[cal["wm_yr_wk"].to_numpy()[:D]].to_numpy(np.float32)
        on_sale = ~np.isnan(price)
        price_norm = np.nan_to_num(price / np.nanmax(price, 1, keepdims=True))
        prev = np.concatenate([price[:, :7], price[:, :-7]], 1)
        price_change = np.clip(np.nan_to_num(np.log(price / prev)), -1, 1)   # raw ratios have outliers up to ~900×

        snap = np.stack([cal[f"snap_{s}"].to_numpy()[:D] for s in sales["state_id"]]).astype(np.float32)
        event = cal["event_name_1"].notna().to_numpy()[:D].astype(np.float32)
        # per-series, per-day real covariates: (N, D, 5)
        self.real = np.stack([price_norm, price_change, on_sale.astype(np.float32), snap,
                              np.broadcast_to(event, (n, D))], -1).astype(np.float32)
        self.wday = (cal["wday"].to_numpy()[:D] - 1).astype(np.int64)
        self.month = (cal["month"].to_numpy()[:D] - 1).astype(np.int64)
        self.static = np.stack([pd.factorize(sales[c])[0] for c in ["item_id", "dept_id", "store_id"]], 1)
        self.first_sale = (self.y > 0).argmax(1)


class DeepARNB(nn.Module):
    def __init__(self, n_items, n_depts, n_stores, n_real=5, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.item = nn.Embedding(n_items, 16)
        self.dept = nn.Embedding(n_depts, 4)
        self.store = nn.Embedding(n_stores, 4)
        self.wday = nn.Embedding(7, 3)
        self.month = nn.Embedding(12, 3)
        n_in = len(LAGS) + 1 + n_real + 16 + 4 + 4 + 3 + 3    # lagged sales, log scale, covariates, embeddings
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 3)

    def inputs(self, lags_scaled, log_v, real, wday, month, static):
        """lags_scaled: (B, T, len(LAGS)) lagged sales divided by the window scale v."""
        T = lags_scaled.shape[1]
        s = torch.cat([self.item(static[:, 0]), self.dept(static[:, 1]), self.store(static[:, 2])], -1)
        return torch.cat([lags_scaled, log_v.expand(-1, T).unsqueeze(-1), real,
                          self.wday(wday), self.month(month), s.unsqueeze(1).expand(-1, T, -1)], -1)

    def forward(self, x, level, state=None):
        """level = context mean of daily sales (0 for windows with no sales)."""
        h, state = self.lstm(x, state)
        out = self.head(h)
        # μ = level × multiplier + small additive term. softplus(0 + 0.5413) = 1, so an untrained model
        # predicts the moving average; the additive term (≈0.05 at init) covers restocks after zero-sales
        # windows without needing a huge multiplier on a near-zero level.
        mu = F.softplus(out[..., 0] + 0.5413) * level + F.softplus(out[..., 2] - 3)
        alpha = F.softplus(out[..., 1]) / (level + 1).sqrt() + 1e-4
        return mu, alpha, state


def window_level(prev_day_ctx):
    """Context mean of previous-day sales. Inputs are divided by (level + 1), as in the DeepAR paper,
    so they stay bounded; dividing by the raw mean produced inputs above 1,000 for sparse series.
    """
    return prev_day_ctx.mean(1, keepdim=True)


def nb_nll(y, mu, alpha):
    """Negative log-likelihood of a negative binomial with mean μ and dispersion α."""
    r = 1 / alpha
    return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
             + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu) + 1e-8))


def nb_sample(mu, alpha):
    r = 1 / alpha
    lam = torch.distributions.Gamma(r, r / mu).sample()
    return torch.poisson(lam)
