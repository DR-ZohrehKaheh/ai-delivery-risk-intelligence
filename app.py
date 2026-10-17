"""Streamlit research dashboard for historical e-commerce delivery risk."""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.input_validation import BRAZIL_STATES, REQUIRED_COLUMNS, validate_batch, spreadsheet_safe_csv

st.set_page_config(page_title="AI Delivery Risk Intelligence", page_icon="📦", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "xgboost_reproduced.joblib"


@st.cache_resource
def load_model(path: str):
    # Only open trusted local model artifacts; joblib deserialization can run code.
    return joblib.load(path)


st.title("AI Delivery Risk Intelligence")
st.caption("Historical e-commerce delivery-delay ranking | Research portfolio prototype")
st.warning(
    "Research demonstration only. Historical Olist data; confirmed delivered orders only. "
    "Model scores are not reliably calibrated and are not validated for live operations."
)

if not MODEL_PATH.is_file():
    st.error("Local model not found. Download Olist CSV files, then run python src/train.py.")
    st.stop()

model = load_model(str(MODEL_PATH))

st.subheader("Single-order assessment")
with st.form("single_order"):
    left, right = st.columns(2)
    with left:
        customer_state = st.selectbox(
            "Customer state", list(BRAZIL_STATES),
            index=list(BRAZIL_STATES).index("SP")
        )
        number_of_items = st.number_input("Number of items", min_value=1, value=1)
        total_price = st.number_input("Total item price (BRL)", min_value=0.0, value=140.0)
        total_freight = st.number_input("Freight cost (BRL)", min_value=0.0, value=22.0)
        promised_days = st.number_input("Promised delivery lead time (days)", min_value=1, value=24)
    with right:
        number_of_sellers = st.number_input(
            "Number of sellers", min_value=1,
            max_value=int(number_of_items), value=1
        )
        number_of_products = st.number_input(
            "Distinct products", min_value=1,
            max_value=int(number_of_items), value=1
        )
        purchase_month = st.selectbox("Purchase month", list(range(1, 13)))
        purchase_weekday = st.selectbox("Purchase weekday (Monday=0)", list(range(7)))
        purchase_hour = st.slider("Purchase hour", 0, 23, 14)
    submitted = st.form_submit_button("Estimate risk score")

if submitted:
    order = pd.DataFrame([{
        "number_of_items": number_of_items,
        "total_price": total_price,
        "total_freight": total_freight,
        "number_of_sellers": number_of_sellers,
        "number_of_products": number_of_products,
        "promised_days": promised_days,
        "customer_state": customer_state,
        "purchase_month": purchase_month,
        "purchase_weekday": purchase_weekday,
        "purchase_hour": purchase_hour,
    }])
    order = validate_batch(order)
    score = float(model.predict_proba(order)[0, 1])
    st.metric("Experimental delivery-delay risk score", f"{score * 100:.2f}%")
    st.progress(score)
    st.caption("Experimental score, not a verified real-world delay probability.")

st.divider()
st.header("Batch risk assessment")
st.write("Upload a CSV of purchase-time order features to rank orders for a limited review budget.")

sample = pd.DataFrame([{
    "number_of_items": 1, "total_price": 140.0, "total_freight": 22.0,
    "number_of_sellers": 1, "number_of_products": 1, "promised_days": 24,
    "customer_state": "SP", "purchase_month": 1, "purchase_weekday": 0,
    "purchase_hour": 14,
}])
st.download_button(
    "Download sample CSV", sample.to_csv(index=False),
    file_name="sample_orders.csv", mime="text/csv"
)
review_capacity = st.slider("Review capacity (%)", min_value=5, max_value=30, value=10, step=5)
upload = st.file_uploader("Upload orders CSV", type=["csv"], max_upload_size=10)

if upload is not None:
    try:
        original = pd.read_csv(upload)
        if len(original) > 100_000:
            raise ValueError("Limit file uploads to 100,000 rows")
        clean = validate_batch(original)
        scores = model.predict_proba(clean)[:, 1]
        ranked = original.copy()
        ranked["Risk Score (%)"] = scores * 100
        ranked = ranked.sort_values(
            "Risk Score (%)", ascending=False, kind="stable"
        ).reset_index(drop=True)

        top_n = max(1, int(np.ceil(review_capacity / 100 * len(ranked))))
        ranked["Review Priority"] = "Normal"
        ranked.loc[:top_n - 1, "Review Priority"] = "Priority Review"

        st.success(f"Assessed {len(ranked):,} orders")
        c1, c2, c3 = st.columns(3)
        c1.metric("Review capacity", f"{review_capacity}%")
        c2.metric("Orders to review", f"{top_n:,}")
        c3.metric("Average model score", f"{ranked['Risk Score (%)'].mean():.2f}%")
        st.dataframe(ranked, width="stretch", hide_index=True)

        st.subheader("Risk score distribution")
        chart = (
            alt.Chart(ranked)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Risk Score (%):Q", bin=alt.Bin(maxbins=8),
                    title="Model score (%)", axis=alt.Axis(format=".1f")
                ),
                y=alt.Y(
                    "count():Q", title="Number of orders",
                    axis=alt.Axis(tickMinStep=1)
                ),
                tooltip=[alt.Tooltip("count():Q", title="Orders")],
            ).properties(height=300)
        )
        st.altair_chart(chart, width="stretch")

        st.download_button(
            "Download ranked results", spreadsheet_safe_csv(ranked),
            file_name="delivery_risk_results.csv", mime="text/csv"
        )
        st.caption(
            "Review flags reflect the top scores within this batch. "
            "They are not validated intervention recommendations."
        )
    except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
        st.error(f"Invalid CSV: {exc}")
    except Exception:
        st.error("Unexpected processing error. Check the uploaded CSV and application logs.")
