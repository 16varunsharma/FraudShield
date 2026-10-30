"""Streamlit UI. Run: streamlit run app/streamlit_app.py  (calls the logic directly, no separate API needed)."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import Transaction, _drivers, _retriever, _score  # noqa: E402
from src.rag import explain  # noqa: E402

st.set_page_config(page_title="FraudShield", layout="wide")
st.title("FraudShield: fraud detection with an AI analyst assistant")

meta_path = ROOT / "models" / "meta.json"
if not meta_path.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()
meta = json.loads(meta_path.read_text())
if meta["data_source"] == "synthetic":
    st.warning("Model trained on SYNTHETIC data (smoke test). Metrics below are not real results.")
else:
    st.caption("Model trained on the public Kaggle (ULB) credit card fraud dataset. Policy documents are samples written for this demo.")

tab1, tab2 = st.tabs(["Score a transaction", "Model performance"])

with tab1:
    demo = pd.read_csv(ROOT / "models" / "demo_samples.csv")
    st.write("Pick a real transaction from the **held-out test set** (the model never saw it in training).")
    labels = [f"#{i}  |  actual: {'FRAUD' if c == 1 else 'legitimate'}  |  amount {a:,.2f}"
              for i, (c, a) in enumerate(zip(demo["Class"], demo["Amount"]))]
    pick = st.selectbox("Transaction", range(len(demo)), format_func=lambda i: labels[i])
    row = demo.drop(columns=["Class"]).iloc[int(pick)].to_dict()
    st.caption(f"True label: {'FRAUD' if demo['Class'].iloc[int(pick)] == 1 else 'legitimate'} "
               "(shown so you can compare with the model's decision)")
    if st.button("Score and explain"):
        tx = Transaction(features={k: float(v) for k, v in row.items()})
        model, m, df_row, prob, flagged = _score(tx)
        drivers = _drivers(model, m, df_row)
        c1, c2 = st.columns(2)
        c1.metric("Fraud probability", f"{prob:.1%}")
        c2.metric("Decision", "FLAGGED" if flagged else "Not flagged", help=f"threshold {m['threshold']:.3f}")
        st.bar_chart(pd.DataFrame(drivers).set_index("feature")["impact"])
        out = explain(_retriever(), prob, flagged, drivers, row["Amount"])
        st.subheader("Analyst assistant")
        st.write(out["explanation"])
        st.caption(f"Generator: {out['engine']} | retrieval: {out['retrieval_backend']}")
        with st.expander("Retrieved policy passages"):
            for s in out["sources"]:
                st.markdown(f"**{s['source']}** (score {s['score']})\n\n{s['text']}")

with tab2:
    res = pd.DataFrame(meta["metrics"]).T[["pr_auc", "roc_auc", "precision", "recall", "est_savings"]]
    st.dataframe(res.style.format("{:.3f}"))
    st.caption(f"Best model: {meta['best_model']} | rows: {meta['n_rows']:,} | fraud rate: {meta['fraud_rate']:.4%}. "
               "Savings use illustrative cost assumptions (see src/train.py).")