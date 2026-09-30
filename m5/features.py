"""Feature table for the global LightGBM model.

Every sales-derived feature is lagged by at least 28 days, so one model forecasts all
28 horizons directly (no recursive feeding of predictions).
"""
import numpy as np
import pandas as pd

from .data import LAST_DAY, load_calendar, load_prices, sales_matrix

MIN_LAG = 28
LAGS = [28, 35, 42, 49, 56]
WINDOWS = [7, 28, 56, 112]
CATEGORICAL = ["item", "dept", "cat", "store", "state", "wday", "month", "event_name", "event_type"]


def _shift(a, k):
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def _rolling_mean(a, w):
    """Mean of a[:, d-w+1 .. d] (NaN until w days are available)."""
    c = np.nancumsum(np.nan_to_num(a), axis=1)
    out = np.full_like(a, np.nan)
    out[:, w - 1 :] = c[:, w - 1 :] - np.concatenate([np.zeros((a.shape[0], 1), a.dtype), c[:, :-w]], axis=1)
    return out / w


def build(sales, first_day):
    """Long-format features for days first_day .. LAST_DAY (target = NaN-free units)."""
    y = sales_matrix(sales)                          # (N, D)
    n, D = y.shape
    cal = load_calendar().set_index("d").loc[1:D]

    # prices on the day grid; NaN = item not on sale yet
    prices = load_prices().pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
    prices = prices.reindex(pd.MultiIndex.from_frame(sales[["store_id", "item_id"]]))
    price = prices[cal["wm_yr_wk"].to_numpy()].to_numpy(np.float32)

    feats = {f"lag_{k}": _shift(y, k) for k in LAGS}
    base = _shift(y, MIN_LAG)
    for w in WINDOWS:
        feats[f"rmean_{w}"] = _rolling_mean(base, w)
    sq = _rolling_mean(base ** 2, 28)
    feats["rstd_28"] = np.sqrt(np.maximum(sq - feats["rmean_28"] ** 2, 0))
    feats["price"] = price
    feats["price_norm"] = price / np.nanmax(price, axis=1, keepdims=True)
    feats["price_change_7"] = price / _shift(price, 7)
    feats["price_vs_mean"] = price / np.nanmean(price, axis=1, keepdims=True)

    first_sale = (y > 0).argmax(1)
    days = np.arange(D)
    feats["days_since_first_sale"] = (days[None, :] - first_sale[:, None]).astype(np.float32)

    state = sales["state_id"].to_numpy()
    snap = np.stack([cal[f"snap_{s}"].to_numpy() for s in state]).astype(np.float32)
    feats["snap"] = snap

    cols = slice(first_day - 1, LAST_DAY)
    keep = (~np.isnan(price[:, cols])).ravel()       # drop days before an item launched in a store
    df = pd.DataFrame({k: v[:, cols].ravel()[keep] for k, v in feats.items()})

    n_days = LAST_DAY - first_day + 1
    df["d"] = np.tile(np.arange(first_day, LAST_DAY + 1), n)[keep]
    df["row"] = np.repeat(np.arange(n), n_days)[keep]
    for col, src in [("item", "item_id"), ("dept", "dept_id"), ("cat", "cat_id"), ("store", "store_id"), ("state", "state_id")]:
        df[col] = pd.factorize(sales[src])[0][df["row"]]
    c = cal.loc[df["d"]]
    df["wday"] = c["wday"].to_numpy()
    df["month"] = c["month"].to_numpy()
    df["mday"] = c["date"].dt.day.to_numpy()
    df["event_name"] = pd.factorize(cal["event_name_1"])[0][df["d"] - 1] + 1
    df["event_type"] = pd.factorize(cal["event_type_1"])[0][df["d"] - 1] + 1
    df["target"] = y[:, cols].ravel()[keep]
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    return df
