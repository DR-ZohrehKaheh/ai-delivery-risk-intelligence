"""Dataset-independent unit tests for the research prototype."""
import unittest

import numpy as np
import pandas as pd

from src.input_validation import REQUIRED_COLUMNS, validate_batch
from src.train import aggregate_order_items, build_features, create_snapshot, make_model


class DeliveryRiskTests(unittest.TestCase):
    def setUp(self):
        self.row = pd.DataFrame([{
            "number_of_items": 1, "total_price": 140.0, "total_freight": 22.0,
            "number_of_sellers": 1, "number_of_products": 1, "promised_days": 24,
            "customer_state": "SP", "purchase_month": 1, "purchase_weekday": 0,
            "purchase_hour": 14,
        }])

    def test_valid_single_and_batch_input(self):
        checked = validate_batch(pd.concat([self.row] * 5, ignore_index=True))
        self.assertEqual(checked.shape, (5, len(REQUIRED_COLUMNS)))

    def test_invalid_batch_is_rejected(self):
        invalid = self.row.copy()
        invalid.loc[0, "number_of_sellers"] = 3
        with self.assertRaises(ValueError):
            validate_batch(invalid)

    def test_temporal_snapshot_excludes_future_outcomes(self):
        orders = pd.DataFrame({
            "order_id": ["a", "b", "c"],
            "order_status": ["delivered", "delivered", "canceled"],
            "order_purchase_timestamp": pd.to_datetime(["2017-10-01"] * 3),
            "order_delivered_customer_date": pd.to_datetime(
                ["2017-10-15", "2018-02-01", None]
            ),
            "order_estimated_delivery_date": pd.to_datetime(["2017-10-10"] * 3),
        })
        snap = create_snapshot(orders, "2017-01-01", "2017-11-01", "2018-01-01")
        self.assertEqual(snap["order_id"].tolist(), ["a"])
        self.assertEqual(snap["late_delivery"].tolist(), [1])

    def test_feature_join_does_not_duplicate_orders(self):
        orders = pd.DataFrame({
            "order_id": ["a"], "customer_id": ["c"],
            "order_purchase_timestamp": pd.to_datetime(["2017-10-01"]),
            "order_estimated_delivery_date": pd.to_datetime(["2017-10-11"]),
            "late_delivery": [0],
        })
        customers = pd.DataFrame({"customer_id": ["c"], "customer_state": ["SP"]})
        items = pd.DataFrame({
            "order_id": ["a", "a"], "order_item_id": [1, 2],
            "price": [10.0, 20.0], "freight_value": [3.0, 4.0],
            "seller_id": ["s", "s"], "product_id": ["x", "y"],
        })
        features = build_features(orders, customers, aggregate_order_items(items))
        self.assertEqual(len(features), 1)
        self.assertEqual(features.loc[0, "number_of_items"], 2)
        self.assertEqual(features.loc[0, "promised_days"], 10)

    def test_model_single_and_batch_inference(self):
        sample = pd.concat([self.row] * 16, ignore_index=True)
        sample["total_price"] = np.linspace(20, 300, 16)
        y = np.array([0, 0, 1, 0] * 4)
        predictor = make_model(n_estimators=6)
        predictor.fit(sample, y)
        one = predictor.predict_proba(self.row)
        five = predictor.predict_proba(pd.concat([self.row] * 5, ignore_index=True))
        self.assertEqual(one.shape, (1, 2))
        self.assertEqual(five.shape, (5, 2))
        self.assertTrue(np.allclose(five[:, 1], one[0, 1]))
        self.assertTrue(np.all((five >= 0) & (five <= 1)))


if __name__ == "__main__":
    unittest.main()
