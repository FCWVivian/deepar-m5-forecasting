# Probabilistic Demand Forecasting: DeepAR, LSTM & Transformer on M5

UC Berkeley capstone project (2023–2024) on deep-learning time-series forecasting. We started by reproducing
**DeepAR** ([Salinas et al., 2017](https://arxiv.org/abs/1704.04110)) on the UCI electricity dataset. Then we adapted
the pipeline to the **M5 Walmart sales** dataset and compared it with an **LSTM** baseline and a **Transformer**
encoder, including a pretrain-then-fine-tune setup.

> This is a personal, cleaned-up copy of a team project. The original team repository, with its full commit history,
> is [miloscola/Capsrone23-DeepAR](https://github.com/miloscola/Capsrone23-DeepAR). See [Team & credits](#team--credits).

![Transformer forecast on an M5 item](assets/transformer_pretrained.png)
*28-day forecast (red) from the pretrained + fine-tuned Transformer on one M5 item (`HOBBIES_1_005_CA_1`).*

## Highlights

- **DeepAR (PyTorch)**: autoregressive LSTM that outputs a Gaussian (μ, σ) at each step, with item embeddings and
  time covariates. Evaluated with ND, RMSE, and ρ-quantile loss (ρ50 / ρ90).
- **M5 adaptation**: `preprocess_M5.py` builds sliding windows from 30,490 item-store series with 6 covariates
  (day of week, month, SNAP CA/TX/WI, and an event flag). It predicts 28 days from a 56-day window.
- **LSTM baseline**: single-series LSTM with a 28-day sliding window and MinMax scaling.
- **Transformer encoder**: a 196-day input window plus covariates, predicting the next 28 days. Experiments cover
  normalization variants (batch/instance norm), attention heads (8 vs 16), and **pretraining** on related series
  (the same item across 4 CA stores) before **fine-tuning** on the target item.

## Repository layout

```
.
├── deepar/                      DeepAR implementation (training scripts)
│   ├── model/net.py             DeepAR network, Gaussian likelihood loss, metrics
│   ├── model/LSTM.py
│   ├── preprocess_elect.py      UCI electricity → train/test windows (.npy)
│   ├── preprocess_M5.py         M5 → train/test windows (.npy)
│   ├── train.py / train_M5.py   training + evaluation loop (electricity / M5)
│   ├── evaluate.py, utils.py, dataloader.py, search_hyperparams.py
│   ├── experiments/             hyperparameter configs (params.json, params_M5.json)
│   └── data -> ../data          symlink
├── notebooks/
│   ├── 01_m5_eda.ipynb                         exploratory data analysis
│   ├── 02_m5_lstm_baseline.ipynb               LSTM baseline
│   ├── 03_m5_transformer.ipynb                 Transformer encoder
│   ├── 04_m5_transformer_batchnorm.ipynb       Transformer + per-window normalization
│   ├── 05_m5_transformer_pretrain.ipynb        pretrain on related series → models/transformer.pth
│   ├── 06_m5_finetune_item4_nhead8.ipynb       fine-tune on target item (8 heads)
│   ├── 07_m5_finetune_item4_nhead16.ipynb      fine-tune on target item (16 heads)
│   ├── 08_m5_transformer_vs_lstm_kaggle.ipynb  Transformer vs LSTM, adapted from a Kaggle notebook
│   ├── variants/                               alternative settings of the notebooks above
│   └── exploration/                            early data-pipeline / feature-engineering drafts
├── data/                        datasets (not committed, see data/README.md)
└── assets/                      figures used in this README
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the datasets as described in [`data/README.md`](data/README.md).

**DeepAR** (run from `deepar/`):

```bash
cd deepar
# Electricity
python preprocess_elect.py
python train.py                      # add --sampling for ancestral sampling
python evaluate.py --restore-file best

# M5
python preprocess_M5.py
python train_M5.py
```

**Notebooks**: open them in Jupyter from inside `notebooks/`. Paths are relative (`../data/M5/`), and the
pretrained Transformer is saved to and loaded from `models/transformer.pth`.

> Notes
> - The DeepAR code was originally written for PyTorch 0.4.1 (see the upstream repo). It hasn't been re-tested on
>   current PyTorch versions yet.
> - Notebooks in `exploration/` are early drafts. They may depend on intermediate files (`data/data_output/...`)
>   written by other drafts.

## Team & credits

- **Vivian Fang** ([@FCWVivian](https://github.com/FCWVivian)): M5 preprocessing and data loader, LSTM baseline,
  Transformer models, pretraining / fine-tuning experiments, EDA
- **[@miloscola](https://github.com/miloscola)**: set up the team repo, DeepAR training/evaluation loop,
  normalization (batch-norm) experiments
- **Gululu**: DeepAR testing
- UC Berkeley Capstone 2023–2024

The DeepAR implementation is based on
[zhykoties/TimeSeries](https://github.com/zhykoties/TimeSeries) by Yunkai Zhang, Qiao Jiang, and Xueying Ma
(Apache-2.0). Some M5 notebooks adapt ideas from public Kaggle notebooks
([davidnguyens16](https://www.kaggle.com/code/davidnguyens16/learning-pytorch-lstm-deep-learning-with-m5-data),
[anshuls235](https://www.kaggle.com/code/anshuls235/time-series-forecasting-eda-fe-modelling)).

## License

Apache License 2.0 (inherited from the upstream DeepAR implementation). See [LICENSE](LICENSE).
