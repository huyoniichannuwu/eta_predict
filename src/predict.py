"""Dự đoán một chuyến bằng cùng cách tạo đặc trưng như lúc học."""
import argparse
import math

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR
from .feature_engineering import FEATURES, add_features


def predict_seconds(model, trips):
    features = add_features(trips)[FEATURES]
    return np.maximum(model.predict(features), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickup_datetime", help="Ngày giờ xuất phát tại New York")
    for name in ("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"):
        parser.add_argument(name, type=float)
    parser.add_argument("--model", choices=["linear_regression", "random_forest", "hgb"], default="linear_regression")
    args = parser.parse_args()
    row = vars(args).copy()
    model_name = row.pop("model")
    try:
        start = pd.Timestamp(row.pop("pickup_datetime"))
        if pd.isna(start):
            raise ValueError()
    except (ValueError, OverflowError):
        parser.error("Ngày giờ xuất phát không hợp lệ.")
    for name, value in row.items():
        bound = 90 if "latitude" in name else 180
        if not math.isfinite(value) or abs(value) > bound:
            parser.error(f"Tọa độ không hợp lệ: {name}")
    trip = pd.DataFrame([{**row, "pickup_datetime": start}])
    model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    seconds = round(predict_seconds(model, trip)[0])
    print(f"{model_name}: {seconds} giây | ETA {start + pd.Timedelta(seconds=seconds)}")


if __name__ == "__main__":
    main()
