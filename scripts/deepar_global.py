"""Train the global DeepAR-NB model and forecast a split with ancestral sampling.

    python scripts/deepar_global.py --split valid     # train on days < 1886, forecast 1886..1913
    python scripts/deepar_global.py --split test      # train on days < 1914, forecast 1914..1941
"""
import argparse
import time

import numpy as np
import torch

from m5.data import HORIZON, SPLITS, load_sales
from m5.deepar_nb import LAGS, Arrays, DeepARNB, nb_nll, nb_sample, window_level
from m5.results import save_forecast

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="valid", choices=list(SPLITS))
parser.add_argument("--steps", type=int, default=12000)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--context", type=int, default=112)
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--decode", default="sample", choices=["sample", "mean"],
                    help="feed back NB samples (DeepAR) or the predicted mean")
parser.add_argument("--load", help="skip training and load these weights")
parser.add_argument("--name", default="deepar_nb")
parser.add_argument("--eval-every", type=int, default=0,
                    help="score the mean-path forecast every N steps (use on the valid split to pick --steps)")
args = parser.parse_args()
if args.decode == "mean":
    args.samples = 1

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

sales = load_sales()
A = Arrays(sales)
start = SPLITS[args.split] - 1            # 0-based index of the first forecast day
C, W = args.context, args.context + HORIZON
MAX_LAG = max(LAGS)

model = DeepARNB(A.static[:, 0].max() + 1, A.static[:, 1].max() + 1, A.static[:, 2].max() + 1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(args.steps, 1))

# windows (plus their lags) must lie after launch and end before the forecast period
earliest_t0 = A.first_sale + MAX_LAG
eligible = np.where(earliest_t0 + W <= start)[0]
volume = A.y[eligible, :start].mean(1) + 1
p_series = volume / volume.sum()          # DeepAR: sample series in proportion to scale


def tensor(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=device)


def batch_tensors(rows, t0, length):
    """Target, lagged sales and covariates for windows rows × [t0, t0+length)."""
    idx = t0[:, None] + np.arange(length)[None, :]
    lags = np.stack([A.y[rows[:, None], idx - k] for k in LAGS], -1)
    return (tensor(A.y[rows[:, None], idx]), tensor(lags), tensor(A.real[rows[:, None], idx]),
            tensor(A.wday[idx], torch.long), tensor(A.month[idx], torch.long), tensor(A.static[rows], torch.long))


def level_of(lags_ctx):
    return window_level(lags_ctx[..., 0])


def encode(lags, level, real, wday, month, static):
    return model.inputs(lags / (level + 1).unsqueeze(-1), torch.log1p(level), real, wday, month, static)


@torch.no_grad()
def forecast(max_rows=512):
    """Warm up on the context, then roll 28 days ahead; return the mean over sample paths.

    Batches are capped at 512 rows (series × samples): on PyTorch 2.2's MPS backend, LSTM outputs are
    silently wrong for large batches (verified: identical rows scored NLL 2.16 in one 4,096-row batch
    vs 1.60 in 512-row chunks).
    """
    model.eval()
    S = args.samples
    chunk = max(1, max_rows // S)
    fut = np.arange(start, start + HORIZON)
    out = np.zeros((len(sales), HORIZON), np.float32)
    for lo in range(0, len(sales), chunk):
        rows = np.arange(lo, min(lo + chunk, len(sales)))
        _, lags, real, wday, month, static = batch_tensors(rows, np.full(len(rows), start - C), C)
        v = level_of(lags)
        _, _, state = model(encode(lags, v, real, wday, month, static), v)

        rep = lambda a: a.repeat_interleave(S, 0)
        state = tuple(s.repeat_interleave(S, 1) for s in state)
        v_s, static_s = rep(v), rep(static)
        # path buffer: the last MAX_LAG observed days, then each generated day appended
        path = rep(tensor(A.y[rows][:, start - MAX_LAG:start]))
        real_f = rep(tensor(A.real[rows][:, fut]))
        wday_f = tensor(A.wday[fut], torch.long).expand(len(rows) * S, -1)
        month_f = tensor(A.month[fut], torch.long).expand(len(rows) * S, -1)
        for h in range(HORIZON):
            lag_now = torch.stack([path[:, -k] for k in LAGS], -1).unsqueeze(1)
            x = encode(lag_now, v_s, real_f[:, h:h + 1], wday_f[:, h:h + 1], month_f[:, h:h + 1], static_s)
            mu, alpha, state = model(x, v_s, state)
            if args.decode == "sample":
                nxt = nb_sample(mu.cpu(), alpha.cpu()).to(device)   # Gamma sampling is CPU-only on MPS
            else:
                nxt = mu
            path = torch.cat([path, nxt], 1)
        out[rows] = path[:, -HORIZON:].reshape(len(rows), S, HORIZON).mean(1).cpu().numpy()
    return out


if args.load:
    model.load_state_dict(torch.load(args.load, map_location=device))
t_start = time.time()
model.train()
for step in range(1, 0 if args.load else args.steps + 1):
    rows = rng.choice(eligible, args.batch, p=p_series)
    t0 = rng.integers(earliest_t0[rows], start - W + 1)
    y, lags, real, wday, month, static = batch_tensors(rows, t0, W)
    v = level_of(lags[:, :C])
    mu, alpha, _ = model(encode(lags, v, real, wday, month, static), v)
    loss = nb_nll(y, mu, alpha).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    opt.step()
    sched.step()
    if step % 500 == 0:
        print(f"step {step:6d}  nll {loss.item():.4f}  {time.time() - t_start:.0f}s", flush=True)
    if args.eval_every and step % args.eval_every == 0:
        samples, decode = args.samples, args.decode
        args.samples, args.decode = 1, "mean"
        from m5.results import evaluator
        print(f"step {step:6d}  valid WRMSSE (mean path) {evaluator(sales, args.split).score(forecast()):.4f}", flush=True)
        args.samples, args.decode = samples, decode
        model.train()


save_forecast(sales, args.name, args.split, forecast())
if not args.load:
    torch.save(model.state_dict(), f"results/{args.name}_{args.split}.pt")
