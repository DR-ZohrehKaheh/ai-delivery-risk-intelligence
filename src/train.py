"""Train and reproduce the historical XGBoost delivery risk experiment.

Run: python src/train.py
Requires Olist CSV files downloaded separately into data/raw/.
Research prototype; not a production prediction system.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"

NUMERIC_FEATURES = [
    "number_of_items", "total_price", "total_freight",
    "number_of_sellers", "number_of_products", "promised_days",
]
CATEGORICAL_FEATURES = [
    "customer_state", "purchase_month", "purchase_weekday", "purchase_hour",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_snapshot(orders: pd.DataFrame, start: str, end: str, cutoff: str) -> pd.DataFrame:
    """Choose confirmed deliveries with a label observed before the cutoff.

    This excludes unresolved orders and may cause selection/censoring bias.
    """
    if not orders["order_id"].is_unique:
        raise ValueError("Order IDs must be unique")
    period = orders.loc[
        (orders["order_purchase_timestamp"] >= pd.Timestamp(start))
        & (orders["order_purchase_timestamp"] < pd.Timestamp(end))
    ].copy()
    delivered_at = period["order_delivered_customer_date"]
    valid = (
        period["order_status"].eq("delivered")
        & delivered_at.notna()
        & period["order_estimated_delivery_date"].notna()
        & delivered_at.lt(pd.Timestamp(cutoff))
    )
    data = period.loc[valid].copy()
    data["late_delivery"] = (
        data["order_delivered_customer_date"].dt.normalize()
        > data["order_estimated_delivery_date"].dt.normalize()
    ).astype(int)
    return data


def aggregate_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple order-item records into one row per order."""
    return (
        items.groupby("order_id")
        .agg(
            number_of_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            number_of_sellers=("seller_id", "nunique"),
            number_of_products=("product_id", "nunique"),
        )
        .reset_index()
    )


def build_features(
    dataset: pd.DataFrame, customers: pd.DataFrame, item_features: pd.DataFrame
) -> pd.DataFrame:
    """Create order features while preserving row count and uniqueness."""
    data = dataset[
        ["order_id", "customer_id", "order_purchase_timestamp",
         "order_estimated_delivery_date", "late_delivery"]
    ].copy()
    data = data.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id", how="left", validate="many_to_one"
    ).merge(
        item_features, on="order_id", how="left", validate="one_to_one"
    )
    data["promised_days"] = (
        data["order_estimated_delivery_date"].dt.normalize()
        - data["order_purchase_timestamp"].dt.normalize()
    ).dt.days
    data["purchase_month"] = data["order_purchase_timestamp"].dt.month
    data["purchase_weekday"] = data["order_purchase_timestamp"].dt.dayofweek
    data["purchase_hour"] = data["order_purchase_timestamp"].dt.hour
    if len(data) != len(dataset) or not data["order_id"].is_unique:
        raise ValueError("Feature join changed row count or order uniqueness")
    if data["promised_days"].isna().any() or (data["promised_days"] < 0).any():
        raise ValueError("Missing or negative promised delivery time")
    return data


def make_model(n_estimators: int = 300) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    classifier = XGBClassifier(
        n_estimators=n_estimators, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=5, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=4,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    required = [
        DATA_DIR / "olist_orders_dataset.csv",
        DATA_DIR / "olist_customers_dataset.csv",
        DATA_DIR / "olist_order_items_dataset.csv",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(
            "Missing Olist CSV files. Download from Kaggle into data/raw/: "
            + ", ".join(missing)
        )
    orders = pd.read_csv(
        required[0],
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                     "order_estimated_delivery_date"],
    )
    customers = pd.read_csv(required[1])
    items = pd.read_csv(required[2])
    item_features = aggregate_order_items(items)

    train = create_snapshot(orders, "2017-01-01", "2017-11-01", "2018-01-01")
    val = create_snapshot(orders, "2018-01-01", "2018-04-01", "2018-07-01")

    train_features = build_features(train, customers, item_features)
    val_features = build_features(val, customers, item_features)
    X_train = train_features[FEATURE_COLUMNS]
    y_train = train_features["late_delivery"]
    X_val = val_features[FEATURE_COLUMNS]
    y_val = val_features["late_delivery"]
    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both classes required in training and validation sets")

    model = make_model()
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    print("\nVALIDATION RESULTS")
    print("Training observations:", len(X_train))
    print("Validation observations:", len(X_val))
    print("ROC-AUC:", round(roc_auc_score(y_val, p_val), 4))
    print("Average Precision:", round(average_precision_score(y_val, p_val), 4))
    print("Brier Score:", round(brier_score_loss(y_val, p_val), 4))
    print("Mean predicted probability (%):", round(p_val.mean() * 100, 2))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "xgboost_reproduced.joblib"
    joblib.dump(model, model_path)
    print("\nTraining completed successfully! Saved model:", model_path)


if __name__ == "__main__":
    main()
