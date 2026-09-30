"""Simple benchmark forecasts, scored with WRMSSE on the validation and test splits."""
import numpy as np

from m5.data import HORIZON, load_sales, split_arrays
from m5.results import save_forecast


def baselines(history):
    return {
        "zeros": np.zeros((len(history), HORIZON)),
        "naive": np.repeat(history[:, -1:], HORIZON, axis=1),
        "seasonal_naive": np.tile(history[:, -7:], HORIZON // 7),
        "moving_avg_28": np.repeat(history[:, -28:].mean(1, keepdims=True), HORIZON, axis=1),
    }


if __name__ == "__main__":
    sales = load_sales()
    for split in ["valid", "test"]:
        history, _ = split_arrays(sales, split)
        for name, forecast in baselines(history).items():
            save_forecast(sales, name, split, forecast)
