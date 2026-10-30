
# FraudShield# FraudShield: fraud detection with an AI analyst assistant

**[Live demo](https://16varunsharma-fraudshield-appstreamlit-app-t72dqc.streamlit.app/)** (Streamlit; the free app may take about a minute to wake up)

An end-to-end machine learning project on highly imbalanced credit card transactions: model comparison,
experiment tracking, per-transaction explanations (SHAP), a FastAPI scoring service, and a retrieval-based
assistant (FAISS) that explains each decision using fraud-policy documents.

![Flagged fraud example](docs/screenshots/fraud_example.png)
![Legitimate example](docs/screenshots/legit_example.png)

> **Honesty notes**
> - Transactions are the public Kaggle *Credit Card Fraud Detection* dataset (ULB): 284,807 transactions, 0.17% fraud. Features V1-V28 are anonymised PCA components.
> - The policy documents in `docs/policies/` are **sample documents I wrote for this demo**, not real company policy.
> - The cost figures used for the "estimated savings" metric are **illustrative assumptions** (see `src/train.py`), not real financial results.
> - This is a demo, not a production system.

## Results (held-out test set)

Data split: stratified train / validation / test. The decision threshold is tuned on the validation set (best F1);
all numbers below come from the untouched test set (about 20% of the data, roughly 98 fraud cases).

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| **XGBoost** (`scale_pos_weight`) | **0.877** | 0.978 | **0.930** | 0.816 |
| Logistic Regression (class-weighted) | 0.712 | 0.974 | 0.784 | 0.816 |
| Isolation Forest (unsupervised baseline) | 0.155 | 0.955 | 0.216 | 0.327 |

XGBoost is selected as the best model (highest PR-AUC). With only about 98 fraud cases in the test set, precision and
recall can move by a few points with a different random split, so treat them as estimates.

## How it works

```
creditcard.csv -> src/train.py   split, imbalance handling, 3 models, MLflow tracking, threshold tuning
               -> models/        model.joblib, meta.json, demo samples
               -> src/api.py     FastAPI: /health, /score, /explain (top drivers per transaction)
               -> src/rag.py     FAISS retrieval over policy docs + optional local LLM (Ollama) or template
               -> app/streamlit_app.py   UI: pick a held-out transaction, score it, see drivers and explanation
```

- **Imbalance:** class weights / `scale_pos_weight` instead of resampling, so validation and test sets keep the real 0.17% fraud rate.
- **Metrics:** PR-AUC, precision and recall rather than accuracy. Predicting "not fraud" every time already gives about 99.8% accuracy.
- **No leakage:** the split happens first; the threshold is chosen on validation only.
- **Explainability:** SHAP values for XGBoost (coefficient contributions for logistic regression). Positive values push the score toward fraud, negative toward legitimate.
- **Assistant (RAG):** policy paragraphs are embedded and indexed with FAISS (cosine similarity). Locally it uses sentence-transformers embeddings
  (`all-MiniLM-L6-v2`) if installed and an Ollama model (`llama3.2`) for the text; otherwise it falls back to TF-IDF retrieval and a deterministic template.
  The hosted demo uses the fallback path. The UI states which engine produced each explanation.
- **Experiment tracking:** every run is logged to MLflow (SQLite backend).

## Run it locally

```bash
pip install -r requirements.txt
# put creditcard.csv (Kaggle, ULB) into data/
python -m src.train                       # trains and saves models/, logs to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q                                 # API tests (use synthetic data in CI)
uvicorn src.api:app --reload              # API docs at http://localhost:8000/docs
streamlit run app/streamlit_app.py        # UI
```

Optional: `pip install sentence-transformers` for embedding search, and run [Ollama](https://ollama.com) with `llama3.2` for LLM-written explanations.
`python -m src.train --synthetic` creates a model on synthetic data for smoke tests only; its metrics are not real results.

A `Dockerfile` for the API and a GitHub Actions workflow (smoke test on synthetic data) are included.

## Limitations

- Anonymised features: explanations name components (V14), not business fields.
- Single public dataset from 2013, no concept-drift monitoring, no real-time or streaming ingestion.
- Scores are not calibrated probabilities (XGBoost with class weighting is overconfident); ranking quality (PR-AUC) is what is evaluated.
- Only about 98 fraud cases in the test set, so metrics have wide uncertainty.
- The RAG assistant is evaluated qualitatively on a small sample policy set, with no formal faithfulness metric.
- Savings estimates depend on assumed costs.

## Next steps

Probability calibration, time-based validation, comparison with SMOTE, a RAG evaluation set (answer faithfulness), drift monitoring on score distribution, and a streaming scoring path.

## Tech

Python, pandas, scikit-learn, XGBoost, SHAP, MLflow, FastAPI, FAISS, sentence-transformers (optional), Ollama (optional), Streamlit, Docker, GitHub Actions.

FraudShield: built fraud detection (XGBoost, Isolation Forest; PR-AUC X, recall Y at Z% precision) on imbalanced transaction data, tracked with MLflow, served via FastAPI and Docker, with a RAG-based assistant (FAISS, LangChain) explaining flags using SHAP and policy documents
=======
# FraudShield: fraud detection with an AI analyst assistant

End-to-end ML project: imbalanced fraud classification, anomaly detection, experiment tracking,
a production-style API, and a RAG assistant that explains flagged transactions using policy documents.

> **Data honesty:** transactions come from the public Kaggle *Credit Card Fraud Detection* dataset (ULB).
> The policy documents in `docs/policies/` are **sample documents written for this demo**.
> Business costs in `src/train.py` are **illustrative assumptions**. This is a deployed demo, not a production system.

## Results
_Fill this in after running on the real dataset (copy from `models/meta.json`):_

| Model | PR-AUC | Recall | Precision |
|---|---|---|---|
| Logistic Regression (class-weighted) | | | |
| XGBoost (scale_pos_weight) | | | |
| Isolation Forest (unsupervised) | | | |

## Architecture
```
creditcard.csv -> src/train.py (split, imbalance handling, 3 models, MLflow, threshold on validation)
              -> models/model.joblib + meta.json
              -> src/api.py   FastAPI: /health /score /explain  (SHAP / linear drivers)
              -> src/rag.py   FAISS retrieval over policy docs + LLM (Ollama) or template fallback
              -> app/streamlit_app.py  UI: score, drivers chart, analyst explanation, metrics
```

## Run it
```bash
pip install -r requirements.txt
# 1. put creditcard.csv (Kaggle, ULB) into data/
python -m src.train                  # trains, logs to MLflow (mlflow ui --backend-store-uri sqlite:///mlflow.db)
pytest -q                            # tests
uvicorn src.api:app --reload         # API docs at http://localhost:8000/docs
streamlit run app/streamlit_app.py   # UI
docker build -t fraudshield . && docker run -p 8000:8000 fraudshield   # needs models/ trained first
```
Smoke test without the dataset: `python -m src.train --synthetic` (metrics are NOT real results; CI uses this).

## Design decisions (be ready to explain these)
- **Metric:** PR-AUC, precision and recall, not accuracy. With ~0.17% fraud, 99.8% accuracy is achievable by predicting "not fraud".
- **Imbalance:** class weights / `scale_pos_weight` instead of oversampling, so the validation and test sets keep the true class ratio.
- **No leakage:** stratified train/validation/test split first; the decision threshold is chosen on validation (best F1); final numbers come from the untouched test set.
- **Cost view:** estimated savings = fraud amount missed with no model minus (missed fraud + review cost per false alert) with the model. Costs are assumptions.
- **Explainability:** SHAP (XGBoost) or coefficient contributions (logistic regression) shown per transaction.
- **RAG:** FAISS cosine search over policy paragraphs; sentence-transformers embeddings when installed, TF-IDF otherwise; answer generated by a local Ollama model if available, else a deterministic template (the UI states which was used).

## Limitations
- Features V1-V28 are anonymised PCA components, so explanations name components, not business fields.
- Single public dataset; no concept-drift monitoring implemented (would track score distribution and precision on reviewed cases).
- Not load-tested; a real deployment would add auth, batching/streaming (Kafka/Spark), a feature store and model monitoring.
- RAG is evaluated qualitatively only on a small sample policy set.

## Next steps
Add calibration plots, SMOTE comparison, cross-validation across time folds, a RAG evaluation set, and drift monitoring.
>>>>>>> 5e4bf43 (FraudShield: fraud detection with RAG analyst assistant)
