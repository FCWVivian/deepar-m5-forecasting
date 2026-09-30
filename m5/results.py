"""Save forecasts and keep a running table of WRMSSE scores in results/metrics.csv."""

import numpy as np
import pandas as pd

from .data import RESULTS_DIR
from .wrmsse import WRMSSE

_evaluators = {}


def evaluator(sales, split):
    if split not in _evaluators:
        _evaluators[split] = WRMSSE(sales, split)
    return _evaluators[split]


def save_forecast(sales, name, split, forecast):
    forecast = np.clip(np.asarray(forecast, dtype=np.float32), 0, None)
    (RESULTS_DIR / "forecasts").mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "forecasts" / f"{name}_{split}.npy", forecast)

    levels = evaluator(sales, split).per_level(forecast)
    row = {"model": name, "split": split, "wrmsse": levels.mean(), **{f"L{i+1}": v for i, v in enumerate(levels)}}
    path = RESULTS_DIR / "metrics.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        table = table[~((table.model == name) & (table.split == split))]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False, float_format="%.4f")
    print(f"{name:20s} {split:5s} WRMSSE {row['wrmsse']:.4f}")
    return row["wrmsse"]
