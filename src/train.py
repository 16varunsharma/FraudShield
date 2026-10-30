"""Train and evaluate fraud models, log to MLflow, save the best model + metadata.

Usage:
    python -m src.train                # real Kaggle data in data/creditcard.csv
    python -m src.train --synthetic    # smoke test only (metrics are NOT real results)
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import FEATURES, TARGET, load_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SEED = 42
# Illustrative business costs (assumptions, change them and say so in the README)
COST_MISSED_FRAUD_MULTIPLIER = 1.0   # missed fraud loses the transaction amount
COST_FALSE_ALERT = 5.0               # review/friction cost per wrongly flagged transaction


def best_f1_threshold(y_true, scores):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.argmax(f1))])


def evaluate(y_true, scores, threshold, amounts):
    pred = (scores >= threshold).astype(int)
    missed = (y_true == 1) & (pred == 0)
    false_alerts = (y_true == 0) & (pred == 1)
    cost_model = COST_MISSED_FRAUD_MULTIPLIER * amounts[missed].sum() + COST_FALSE_ALERT * false_alerts.sum()
    cost_no_model = COST_MISSED_FRAUD_MULTIPLIER * amounts[y_true == 1].sum()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "threshold": float(threshold),
        "cost_with_model": float(cost_model),
        "cost_without_model": float(cost_no_model),
        "est_savings": float(cost_no_model - cost_model),
    }


def main(synthetic: bool):
    df, source = load_data(allow_synthetic=synthetic)
    X, y = df[FEATURES], df[TARGET].values
    # Stratified split; threshold chosen on validation, final numbers on untouched test set
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr",
                                 random_state=SEED, n_jobs=-1),
    }
    mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR.parent / 'mlflow.db'}")
    mlflow.set_experiment("fraudshield")
    results, fitted = {}, {}

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)
            thr = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
            m = evaluate(y_te, model.predict_proba(X_te)[:, 1], thr, X_te["Amount"].values)
            mlflow.log_params({"model": name, "data_source": source, "seed": SEED})
            mlflow.log_metrics({k: v for k, v in m.items()})
            results[name], fitted[name] = m, model
            print(f"{name:8s} PR-AUC={m['pr_auc']:.4f} recall={m['recall']:.3f} precision={m['precision']:.3f}")

    # Unsupervised baseline: Isolation Forest (anomaly detection, no labels used in fit)
    with mlflow.start_run(run_name="isolation_forest"):
        iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()), random_state=SEED, n_jobs=-1)
        iso.fit(X_tr)
        s_val, s_te = -iso.score_samples(X_val), -iso.score_samples(X_te)
        thr = best_f1_threshold(y_val, s_val)
        m = evaluate(y_te, s_te, thr, X_te["Amount"].values)
        mlflow.log_params({"model": "isolation_forest", "data_source": source})
        mlflow.log_metrics(m)
        results["isolation_forest"] = m
        print(f"isoforest PR-AUC={m['pr_auc']:.4f} recall={m['recall']:.3f} precision={m['precision']:.3f}")

    best = max(["logreg", "xgboost"], key=lambda k: results[k]["pr_auc"])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODELS_DIR / "model.joblib")
    # small background sample for SHAP / feature statistics
    X_tr.sample(min(200, len(X_tr)), random_state=SEED).to_csv(MODELS_DIR / "background.csv", index=False)
    # Demo samples come from the held-out TEST set (never used for training or threshold tuning)
    te = X_te.copy()
    te[TARGET] = y_te
    demo = pd.concat([te[te[TARGET] == 1].sample(min(10, int(y_te.sum())), random_state=SEED),
                      te[te[TARGET] == 0].sample(10, random_state=SEED)])
    demo.to_csv(MODELS_DIR / "demo_samples.csv", index=False)
    meta = {
        "best_model": best, "features": FEATURES, "threshold": results[best]["threshold"],
        "data_source": source, "n_rows": int(len(df)), "fraud_rate": float(y.mean()),
        "metrics": results,
        "note": "SYNTHETIC DATA - NOT REAL RESULTS" if source == "synthetic" else "Kaggle ULB credit card fraud data",
    }
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nBest model: {best}  | data source: {source}\nSaved to {MODELS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="smoke test on synthetic data only")
    main(ap.parse_args().synthetic)