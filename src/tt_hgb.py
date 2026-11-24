"""Minh họa các cây HGB trên một chuyến trong tập kiểm thử đã chia theo thời gian."""
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR


def find_trip(path, trip_id):
    for chunk in pd.read_csv(path, chunksize=25000):
        match = chunk.loc[chunk["id"].eq(trip_id)]
        if not match.empty:
            return match
    raise ValueError(f"Không tìm thấy {trip_id} trong {path.name}")


def first_tree_path(model, X):
    """Đọc các nút nội bộ của sklearn cho mô hình dùng đặc trưng số của dự án."""
    nodes = model._predictors[0][0].nodes
    index = 0
    path = []
    while not nodes[index]["is_leaf"]:
        node = nodes[index]
        feature = X.columns[node["feature_idx"]]
        value = X[feature].item()
        threshold = float(node["num_threshold"])
        left = bool(node["missing_go_to_left"]) if pd.isna(value) else value <= threshold
        child = int(node["left"] if left else node["right"])
        path.append((index, feature, value, threshold, "trái" if left else "phải", child))
        index = child
    return pd.DataFrame(path, columns=["nút", "đặc trưng", "giá trị", "ngưỡng", "nhánh", "nút sau"]), index, float(nodes[index]["value"])


def stage_history(model, X):
    # Với squared_error, vòng 0 của mô hình này là trung bình nhãn huấn luyện.
    baseline = model._baseline_prediction.item()
    seconds = [baseline] + [stage.item() for stage in model.staged_predict(X)]
    history = pd.DataFrame({"dự đoán (s)": seconds}).rename_axis("vòng")
    history["điều chỉnh (s)"] = history["dự đoán (s)"].diff()
    np.testing.assert_allclose(seconds[-1], model.predict(X).item(), rtol=0, atol=1e-8)
    return history


def plot_history(history, observed, trip_id):
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.plot(history.index, history["dự đoán (s)"], color="#146a68", label="Dự đoán sau mỗi vòng")
    ax.axhline(observed, color="#a24525", linestyle="--", label=f"Thực tế ({observed} giây)")
    ax.set(xlabel="Vòng boosting (0 là dự đoán ban đầu)", ylabel="Thời lượng (giây)",
           title=f"Diễn biến dự đoán HGB cho chuyến {trip_id}")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_id", nargs="?", default="id1459588")
    args = parser.parse_args()
    model = joblib.load(MODELS_DIR / "hgb.joblib")
    trip = find_trip(PROCESSED_DIR / "test.csv", args.trip_id)
    X = trip.loc[:, model.feature_names_in_]
    path, leaf_id, delta = first_tree_path(model, X)
    history = stage_history(model, X)
    np.testing.assert_allclose(delta, history.loc[1, "điều chỉnh (s)"], rtol=0, atol=1e-8)
    print(path.to_string(index=False))
    print(f"Lá {leaf_id} điều chỉnh {delta:+.3f} giây")
    print(history.loc[[0, 1, 2, 3, model.n_iter_]].round(3))
    fig = plot_history(history, trip["trip_duration"].item(), args.trip_id)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    name = "hinh_hgb_200_vong.png" if args.trip_id == "id1459588" and model.n_iter_ == 200 else f"hgb_{args.trip_id}_{model.n_iter_}.png"
    fig.savefig(FIGURES_DIR / name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
