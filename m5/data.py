"""Loading helpers and split definitions for M5 (sales_train_evaluation: d_1 .. d_1941)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "M5"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

HORIZON = 28
LAST_DAY = 1941
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Each split forecasts 28 days; everything before `start` is history.
#   test = the M5 public-leaderboard period, valid = the 28 days before it.
SPLITS = {
    "valid": LAST_DAY - 2 * HORIZON + 1,  # d_1886 .. d_1913
    "test": LAST_DAY - HORIZON + 1,       # d_1914 .. d_1941
}


def load_sales():
    return pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")


def load_calendar():
    cal = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    cal["d"] = cal["d"].str[2:].astype(int)
    return cal


def load_prices():
    return pd.read_csv(DATA_DIR / "sell_prices.csv")


def day_cols(first, last):
    return [f"d_{d}" for d in range(first, last + 1)]


def sales_matrix(sales):
    """(30490, 1941) float32 array of daily unit sales."""
    return sales[day_cols(1, LAST_DAY)].to_numpy(np.float32)


def split_arrays(sales, split):
    """History (all days before the split) and the 28-day target for a split."""
    y = sales_matrix(sales)
    start = SPLITS[split]
    return y[:, : start - 1], y[:, start - 1 : start - 1 + HORIZON]
