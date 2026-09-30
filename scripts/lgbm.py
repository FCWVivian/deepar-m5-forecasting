"""Global LightGBM model with a Tweedie objective (suits intermittent, zero-heavy counts).

valid: train on days < 1886, early-stop on d_1886..1913.
test:  retrain on days < 1914 with the best iteration count, forecast d_1914..1941.
"""
import argparse

import lightgbm as lgb
import numpy as np

from m5.data import HORIZON, SPLITS, load_sales
from m5.features import CATEGORICAL, build
from m5.results import save_forecast

parser = argparse.ArgumentParser()
parser.add_argument("--first-day", type=int, default=1100, help="first day used for training")
parser.add_argument("--loss", default="tweedie", choices=["tweedie", "l2"])
args = parser.parse_args()

PARAMS = dict(
    objective=args.loss,
    tweedie_variance_power=1.1,
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=0.1,
    verbose=-1,
    num_threads=10,
)
NAME = "lightgbm_tweedie" if args.loss == "tweedie" else "lightgbm_l2"

sales = load_sales()
df = build(sales, args.first_day)
features = [c for c in df.columns if c not in ("d", "row", "target")]
print(f"{len(df):,} rows × {len(features)} features")


def forecast(model, start):
    part = df[(df.d >= start) & (df.d < start + HORIZON)]
    out = np.zeros((len(sales), HORIZON), np.float32)   # rows not yet on sale stay 0
    out[part.row.to_numpy(), part.d.to_numpy() - start] = model.predict(part[features])
    return out


def dataset(mask):
    return lgb.Dataset(df.loc[mask, features], df.loc[mask, "target"], categorical_feature=CATEGORICAL, free_raw_data=True)


v = SPLITS["valid"]
train = dataset(df.d < v)
valid = dataset((df.d >= v) & (df.d < v + HORIZON))
model = lgb.train(PARAMS, train, num_boost_round=3000, valid_sets=[valid],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
save_forecast(sales, NAME, "valid", forecast(model, v))

t = SPLITS["test"]
model = lgb.train(PARAMS, dataset(df.d < t), num_boost_round=model.best_iteration)
save_forecast(sales, NAME, "test", forecast(model, t))
model.save_model(f"results/{NAME}.txt")

importance = sorted(zip(model.feature_importance("gain"), features), reverse=True)
print("top features:", ", ".join(f for _, f in importance[:10]))
