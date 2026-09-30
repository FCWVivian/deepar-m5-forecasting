"""WRMSSE, the official M5 accuracy metric, over all 12 aggregation levels (42,840 series).

For each series:  RMSSE = sqrt( mean((y - ŷ)²) / mean(Δy_train²) ),
where the scale uses the history from the series' first non-zero sale onward.
Weights are each series' dollar sales over the last 28 days of history, normalized per level;
the final score is the average of the 12 per-level weighted sums.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .data import HORIZON, SPLITS, load_calendar, load_prices, split_arrays

LEVELS = [
    [],                        # 1  total
    ["state_id"],              # 2
    ["store_id"],              # 3
    ["cat_id"],                # 4
    ["dept_id"],               # 5
    ["state_id", "cat_id"],    # 6
    ["state_id", "dept_id"],   # 7
    ["store_id", "cat_id"],    # 8
    ["store_id", "dept_id"],   # 9
    ["item_id"],               # 10
    ["item_id", "state_id"],   # 11
    ["item_id", "store_id"],   # 12 (the 30,490 bottom-level series)
]


def _aggregation_matrix(sales):
    """Sparse (42840, 30490) 0/1 matrix mapping bottom series to every aggregate."""
    blocks, level_of_row = [], []
    n = len(sales)
    for lvl, cols in enumerate(LEVELS):
        if cols:
            codes, _ = pd.factorize(sales[cols].astype(str).agg("|".join, axis=1))
        else:
            codes = np.zeros(n, dtype=int)
        m = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(codes.max() + 1, n))
        blocks.append(m)
        level_of_row += [lvl] * m.shape[0]
    return sparse.vstack(blocks).tocsr(), np.array(level_of_row)


class WRMSSE:
    def __init__(self, sales, split):
        self.split = split
        history, self.target = split_arrays(sales, split)
        self.agg, self.level = _aggregation_matrix(sales)

        hist_agg = self.agg @ history
        started = np.cumsum(hist_agg != 0, axis=1) > 0
        diffs = np.diff(hist_agg, axis=1) ** 2
        valid = started[:, 1:] & started[:, :-1]
        self.scale = (diffs * valid).sum(1) / np.maximum(valid.sum(1), 1)

        dollars = self._last28_dollars(sales, history)
        w = self.agg @ dollars
        for lvl in range(len(LEVELS)):
            rows = self.level == lvl
            w[rows] /= w[rows].sum()
        self.weights = w
        self.target_agg = self.agg @ self.target

    def _last28_dollars(self, sales, history):
        cal = load_calendar().set_index("d")["wm_yr_wk"]
        start = SPLITS[self.split]
        days = np.arange(start - HORIZON, start)
        prices = load_prices()
        wide = prices.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
        wide = wide.reindex(pd.MultiIndex.from_frame(sales[["store_id", "item_id"]]))
        p = wide[cal.loc[days].to_numpy()].to_numpy()
        units = history[:, -HORIZON:]
        return np.nansum(units * p, axis=1)

    def per_level(self, forecast):
        """RMSSE-weighted sum for each of the 12 levels."""
        f_agg = self.agg @ np.asarray(forecast, dtype=np.float64)
        mse = ((self.target_agg - f_agg) ** 2).mean(1)
        rmsse = np.sqrt(mse / self.scale)
        return np.array([(self.weights * rmsse)[self.level == l].sum() for l in range(len(LEVELS))])

    def score(self, forecast):
        return self.per_level(forecast).mean()
