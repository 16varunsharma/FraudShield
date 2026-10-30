"""Data loading.

Real data: Kaggle "Credit Card Fraud Detection" (ULB) -> put creditcard.csv in data/.
Columns: Time, V1..V28 (PCA-anonymised), Amount, Class (1 = fraud).

A synthetic generator is included ONLY so the pipeline can be smoke-tested without
the Kaggle file (CI, quick checks). Never report synthetic-data metrics as results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def make_synthetic(n: int = 40000, fraud_rate: float = 0.004, seed: int = 42) -> pd.DataFrame:
    """Synthetic data with the same schema as the Kaggle file. FOR TESTING ONLY."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 28))
    y = (rng.random(n) < fraud_rate).astype(int)
    shift = np.zeros(28)
    shift[[2, 3, 9, 11, 13, 16]] = [-2.0, 1.8, -2.2, -2.5, -2.0, -1.5]
    X[y == 1] += shift + rng.normal(scale=0.5, size=(int(y.sum()), 28))
    amount = np.where(y == 1, rng.lognormal(4.5, 1.2, n), rng.lognormal(3.5, 1.1, n))
    t = np.sort(rng.integers(0, 172800, n))
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", t)
    df["Amount"] = amount.round(2)
    df[TARGET] = y
    return df


def load_data(allow_synthetic: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source). source is 'kaggle' or 'synthetic'."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH), "kaggle"
    if allow_synthetic:
        return make_synthetic(), "synthetic"
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Download 'creditcard.csv' from Kaggle "
        "(Credit Card Fraud Detection by ULB) and place it in data/. "
        "Use --synthetic only for smoke tests."
    )
