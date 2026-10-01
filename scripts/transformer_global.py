"""Train the global Transformer (direct 28-day negative-binomial forecast) and score a split.

    python scripts/transformer_global.py --split valid
    python scripts/transformer_global.py --split test
"""
import argparse
import time

import numpy as np
import torch

from m5.data import HORIZON, SPLITS, load_sales
from m5.deepar_nb import Arrays, nb_nll
from m5.results import save_forecast
from m5.transformer import TransformerNB

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="valid", choices=list(SPLITS))
parser.add_argument("--steps", type=int, default=12000)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--context", type=int, default=112)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--name", default="transformer_nb")
args = parser.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

sales = load_sales()
A = Arrays(sales)
start = SPLITS[args.split] - 1
C, W = args.context, args.context + HORIZON

model = TransformerNB(A.static[:, 0].max() + 1, A.static[:, 1].max() + 1, A.static[:, 2].max() + 1,
                      context=C, horizon=HORIZON).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

earliest_t0 = A.first_sale + 1
eligible = np.where(earliest_t0 + W <= start)[0]
volume = A.y[eligible, :start].mean(1) + 1
p_series = volume / volume.sum()          # same window sampling as DeepAR


def tensor(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=device)


def batch_tensors(rows, t0):
    """Windows rows × [t0, t0 + C + 28): past sales, level, covariates for all days, future targets."""
    idx = t0[:, None] + np.arange(W)[None, :]
    y = tensor(A.y[rows[:, None], idx])
    level = y[:, :C].mean(1, keepdim=True)
    return (y[:, :C] / (level + 1), level, tensor(A.real[rows[:, None], idx]), tensor(A.wday[idx], torch.long),
            tensor(A.month[idx], torch.long), tensor(A.static[rows], torch.long)), y[:, C:]


t_start = time.time()
model.train()
for step in range(1, args.steps + 1):
    rows = rng.choice(eligible, args.batch, p=p_series)
    t0 = rng.integers(earliest_t0[rows], start - W + 1)
    inputs, target = batch_tensors(rows, t0)
    mu, alpha = model(*inputs)
    loss = nb_nll(target, mu, alpha).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 500 == 0:
        print(f"step {step:6d}  nll {loss.item():.4f}  {time.time() - t_start:.0f}s", flush=True)


@torch.no_grad()
def forecast(chunk=512):
    """Direct forecast: the NB mean for each of the 28 days (no autoregressive rollout needed)."""
    model.eval()
    out = np.zeros((len(sales), HORIZON), np.float32)
    for lo in range(0, len(sales), chunk):
        rows = np.arange(lo, min(lo + chunk, len(sales)))
        # the target slice past LAST_DAY is never used; covariates for the forecast days are known
        inputs, _ = batch_tensors(rows, np.full(len(rows), start - C))
        mu, _ = model(*inputs)
        out[rows] = mu.cpu().numpy()
    return out


save_forecast(sales, args.name, args.split, forecast())
torch.save(model.state_dict(), f"results/{args.name}_{args.split}.pt")
