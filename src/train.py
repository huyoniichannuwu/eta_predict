"""Huấn luyện hai mô hình và tính MAE trên cùng tập kiểm thử theo thời gian."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error        #mae, rmse

from .config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, ROOT_DIR
from .data_pipeline import load_split
from .feature_engineering import FEATURES, add_features
from .predict import predict_seconds


def main():
    train, test = load_split(split_mode="time")
    cutoff = test["pickup_datetime"].min()
    train = train.loc[pd.to_datetime(train["dropoff_datetime"]) < cutoff]
    train_features = add_features(train)
    test_features = add_features(test)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reports = ROOT_DIR / "reports"
    reports.mkdir(exist_ok=True)

    for name, frame in (("train", train_features), ("test", test_features)):
        frame[["id", *FEATURES, "trip_duration"]].to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=RANDOM_SEED),
        "hgb": HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.1, max_iter=200,max_leaf_nodes=31, early_stopping=False, random_state=RANDOM_SEED),
    }
    
    predictions = test_features[["id", "pickup_datetime", "is_rush_hour", "trip_duration"]].copy()
    groups = {"overall": np.ones(len(test), dtype=bool),
              "peak": predictions.is_rush_hour == 1, "off_peak": predictions.is_rush_hour == 0}
    scores = []
    for name, model in models.items():
        #hoc
        model.fit(train_features[FEATURES], train["trip_duration"])
        
        pred = predict_seconds(model, test)
        predictions[name] = pred
        
        for group, mask in groups.items():
            #chi so mae, rmse
            y_true = test.loc[mask, "trip_duration"]
            y_pred = pred[mask]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)

            scores.append({"model": name, "group": group, "n": int(mask.sum()), "mae_minutes": mae/60, "rmse_minutes": rmse/60})
        joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)
    
    metrics = pd.DataFrame(scores)
    
    metrics.to_csv(reports / "metrics.csv", index=False)
    predictions.to_csv(reports / "predictions.csv", index=False)
    
    print(f"Train {len(train)}, test {len(test)}, cutoff {cutoff}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
