"""README figures and the single-item comparison with the original capstone Transformer."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m5.data import HORIZON, RESULTS_DIR, SPLITS, load_sales, sales_matrix

ASSETS = RESULTS_DIR.parent / "assets"
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e1"
SERIES = {"lightgbm_tweedie": ("LightGBM (Tweedie)", "#2a78d6"), "deepar_nb": ("DeepAR (neg. binomial)", "#eb6834"),
          "transformer_nb": ("Transformer (fixed, global)", "#1baf7a")}
LABELS = {"zeros": "All zeros", "naive": "Naive (last day)", "seasonal_naive": "Seasonal naive",
          "moving_avg_28": "28-day moving average", "lightgbm_l2": "LightGBM (MSE loss)",
          **{k: v[0] for k, v in SERIES.items()}}

plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
test = metrics[(metrics.split == "test") & ~metrics.model.isin(["zeros", "deepar_nb_seed1"])].sort_values("wrmsse", ascending=False)

# 1. test WRMSSE by model
fig, ax = plt.subplots(figsize=(8, 0.5 * len(test) + 1))
family = lambda m: "lightgbm_tweedie" if m.startswith("lightgbm") else m   # both LightGBM losses share a color
colors = [SERIES[family(m)][1] if family(m) in SERIES else "#c3c2b7" for m in test.model]
bar_label = lambda m: "DeepAR (neg. binomial, seed 0)" if m == "deepar_nb" else LABELS[m]
ax.barh([bar_label(m) for m in test.model], test.wrmsse, color=colors, height=0.6)
for y, v in enumerate(test.wrmsse):
    ax.text(v + 0.01, y, f"{v:.3f}", va="center", color=INK)
ax.set_xlabel("WRMSSE on d_1914–1941 (lower is better)")
ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.set_xlim(0, test.wrmsse.max() * 1.15)
fig.tight_layout()
fig.savefig(ASSETS / "wrmsse_test.png", dpi=150)

# 2. forecasts: total sales and the capstone's example item
sales = load_sales()
y = sales_matrix(sales)
start = SPLITS["test"] - 1
fc = {m: np.load(RESULTS_DIR / "forecasts" / f"{m}_test.npy") for m in SERIES}
item = np.where(sales.id == "HOBBIES_1_005_CA_1_evaluation")[0][0]          # the capstone's example item
top = y[:, start - 28:start].sum(1).argmax()                                  # best seller, chosen by volume
top_name = sales.id.values[top].replace("_evaluation", "")
panels = [("All 30,490 series summed", lambda a: a.sum(0)), (f"{top_name} (best-selling item-store series)", lambda a: a[top])]

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
days = np.arange(start - 56, start + HORIZON) + 1
for ax, (title, pick) in zip(axes, panels):
    ax.plot(days, pick(y[:, start - 56:start + HORIZON]), color=INK, lw=1.5, label="Actual")
    for m, (label, color) in SERIES.items():
        ax.plot(days[-HORIZON:], pick(fc[m]), color=color, lw=2, label=label)
    ax.axvline(start + 0.5, color=MUTED, ls="--", lw=1)
    ax.set_title(title, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID)
    ax.set_ylabel("Units sold")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels))
axes[1].set_xlabel("Day (d_)")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(ASSETS / "forecasts_test.png", dpi=150)

# 3. single-item RMSE, same item and 28 days as the capstone notebooks
actual = y[item, start:start + HORIZON]
rows = {"Transformer, pretrained + fine-tuned (capstone)": 1.2406}
for m in ["moving_avg_28", "seasonal_naive", *SERIES]:
    if not (RESULTS_DIR / "forecasts" / f"{m}_test.npy").exists():
        continue
    f = np.load(RESULTS_DIR / "forecasts" / f"{m}_test.npy")[item]
    rows[LABELS[m]] = float(np.sqrt(((actual - f) ** 2).mean()))
print("| Model | RMSE |\n|---|---|")
for k, v in sorted(rows.items(), key=lambda kv: kv[1]):
    print(f"| {k} | {v:.2f} |")
