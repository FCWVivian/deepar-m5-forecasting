# Data

Datasets are not committed to this repo. Download them into this folder:

```
data/
├── M5/
│   ├── calendar.csv
│   ├── sales_train_evaluation.csv
│   ├── sales_train_validation.csv   # only needed by 08_m5_transformer_vs_lstm_kaggle
│   ├── sell_prices.csv
│   └── sample_submission.csv        # only needed by some notebooks
└── elect/                           # created automatically by deepar/preprocess_elect.py
```

- **M5 Forecasting (Walmart)**: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
  ```bash
  kaggle competitions download -c m5-forecasting-accuracy -p data/M5 && unzip data/M5/*.zip -d data/M5
  ```
- **Electricity (UCI ElectricityLoadDiagrams20112014)**: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
  Downloaded automatically by `python preprocess_elect.py` (run from `deepar/`).

`deepar/data` is a symlink to this folder, so the DeepAR scripts and the notebooks share the same data.
