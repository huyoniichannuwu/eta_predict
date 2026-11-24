"""Chạy từ thư mục eta_predict"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def so(value, digits=1):
    return f"{value:,.{digits}f}".translate(str.maketrans(",.", ".,"))


def find_trip(path, trip_id):
    for block in pd.read_csv(path, chunksize=25_000):
        match = block.loc[block["id"] == trip_id]
        if not match.empty:
            return match.iloc[[0]]
    raise ValueError(f"Không tìm thấy {trip_id} trong {path}")


def draw_path(tree, x, trip_id, output):
    structure = tree.tree_
    values = x.to_numpy(dtype=np.float32)
    node = 0
    decisions = []
    while structure.children_left[node] != structure.children_right[node]:
        feature = structure.feature[node]
        value = values[0, feature]
        threshold = structure.threshold[node]
        left = float(value) <= threshold
        decisions.append((node, x.columns[feature], value, threshold, left))
        node = structure.children_left[node] if left else structure.children_right[node]
    assert node == tree.apply(values)[0]
    leaf_mean = structure.value[node].item()
    assert np.isclose(leaf_mean, tree.predict(values)[0])

    fig, ax = plt.subplots(figsize=(9.4, 5.35))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.04, .94, "Một chuyến đi qua cây quyết định số 1", fontsize=20, weight="bold")
    ax.text(.04, .865, f"Chuyến {trip_id}  ·  Ba lần rẽ đầu tiên trên đường đến lá", color="#334155")
    for i, (idx, name, val, cut, left) in enumerate(decisions[:3]):
        y = .71 - i * .15
        ax.text(.04, y, f"Nút {idx}", weight="bold", color="#12645c")
        sign = "≤" if left else ">"
        direction = "trái" if left else "phải"
        unit = " km" if name.endswith("_km") else ""
        ax.text(.23, y, f"{name}   {so(val, 4)}{unit}   {sign}   {so(cut, 4)}{unit}", fontsize=12)
        ax.text(.82, y, f"Rẽ {direction}", weight="bold")
        ax.plot([.04, .96], [y-.045]*2, color="#dae2e6", lw=1)
    ax.text(.04, .265, f"Đi tiếp qua {len(decisions)-3} nút theo cùng quy tắc.", color="#475569")
    ax.text(.04, .135, f"Lá {node}   →   Cây 1 dự đoán {so(leaf_mean, 1)} giây",
            fontsize=15, weight="bold", color="#12645c")
    ax.text(.04, .05, "Ngưỡng hiển thị được làm tròn. Python so sánh bằng ngưỡng lưu trong cây.", fontsize=10)
    fig.savefig(output / "hinh_1_duong_di_cay.png", dpi=210, bbox_inches="tight")
    plt.close(fig)
    for idx, name, val, cut, left in decisions[:3]:
        print(f"Node {idx}: {name} = {val:.6f}, threshold = {cut:.6f},", "left" if left else "right")
    print(f"Leaf {node}: {leaf_mean:.6f} s ({len(decisions)} splits)")


def draw_forest(predictions, actual, trip_id, output):
    mean = predictions.mean()
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    fig.subplots_adjust(left=.12, right=.97, top=.76, bottom=.17)
    fig.text(.055, .93, f"Dự đoán của {len(predictions)} cây cho chuyến {trip_id}", fontsize=18, weight="bold")
    fig.text(.055, .85, "Mỗi chấm là một cây. Thứ tự cây theo mô hình đã lưu.", color="#334155")
    ax.scatter(np.arange(1, len(predictions)+1), predictions, s=22, color="#46678a", alpha=.82)
    ax.axhline(mean, color="#12645c", lw=2.2, label=f"Trung bình rừng   {so(mean, 3)} giây")
    ax.axhline(actual, color="#9b471b", lw=1.8, ls="--", label=f"Thực tế   {so(actual, 0)} giây")
    ax.set(xlabel="Số thứ tự cây", ylabel="Thời lượng (giây)", xlim=(0, len(predictions)+1))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1), ncol=2)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "hinh_2_du_doan_100_cay.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    trip_id = "id1459588"
    model = joblib.load(args.project / "models/random_forest.joblib")
    row = find_trip(args.project / "data/processed/test.csv", trip_id)
    x = row.loc[:, model.feature_names_in_]
    actual = row["trip_duration"].item()

    # Mỗi cây nhận cùng bảy giá trị của chuyến đang xét.
    predictions = np.array([tree.predict(x.to_numpy())[0] for tree in model.estimators_])
    forest_prediction = model.predict(x)[0]
    np.testing.assert_allclose(predictions.mean(), forest_prediction, rtol=0, atol=1e-8)
    args.out.mkdir(parents=True, exist_ok=True)
    draw_path(model.estimators_[0], x, trip_id, args.out)
    draw_forest(predictions, actual, trip_id, args.out)
    print(x.to_string(index=False))
    print(f"trees={len(predictions)}, bootstrap={model.bootstrap}, min_samples_leaf={model.min_samples_leaf}, seed={model.random_state}")
    print("First three trees (s):", predictions[:3])
    print(f"Mean of trees = {predictions.mean():.9f} s; model.predict = {forest_prediction:.9f} s")
    print(f"Observed = {actual} s; absolute error = {abs(actual - forest_prediction):.6f} s")


if __name__ == "__main__":
    main()
