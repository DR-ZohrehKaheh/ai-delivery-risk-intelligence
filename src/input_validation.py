"""Validation helpers used by the Streamlit app and automated tests."""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "number_of_items", "total_price", "total_freight",
    "number_of_sellers", "number_of_products", "promised_days",
    "customer_state", "purchase_month", "purchase_weekday", "purchase_hour",
]
BRAZIL_STATES = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
)


def validate_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return data in the exact feature order the model expects."""
    missing = [col for col in REQUIRED_COLUMNS if col not in frame]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Uploaded file has no rows")
    data = frame[REQUIRED_COLUMNS].copy()
    data["customer_state"] = data["customer_state"].astype("string").str.strip().str.upper()
    if not data["customer_state"].isin(BRAZIL_STATES).all():
        raise ValueError("Customer state must be a two-letter Brazilian state code")
    numeric = [col for col in REQUIRED_COLUMNS if col != "customer_state"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Numerical fields must be present and finite")
    integer_cols = [
        "number_of_items", "number_of_sellers", "number_of_products",
        "promised_days", "purchase_month", "purchase_weekday", "purchase_hour",
    ]
    for col in integer_cols:
        if (data[col] != data[col].round()).any():
            raise ValueError(f"{col} must contain whole numbers")
        data[col] = data[col].astype(int)
    if (
        (data[["number_of_items", "number_of_sellers", "number_of_products", "promised_days"]] < 1).any().any()
        or (data[["total_price", "total_freight"]] < 0).any().any()
    ):
        raise ValueError("Counts/lead times must be positive; price/freight must be nonnegative")
    if (
        (data["number_of_sellers"] > data["number_of_items"]).any()
        or (data["number_of_products"] > data["number_of_items"]).any()
    ):
        raise ValueError("Seller/product counts cannot exceed item counts")
    if not (
        data["purchase_month"].between(1, 12).all()
        and data["purchase_weekday"].between(0, 6).all()
        and data["purchase_hour"].between(0, 23).all()
    ):
        raise ValueError("Invalid purchase month, weekday, or hour")
    return data
