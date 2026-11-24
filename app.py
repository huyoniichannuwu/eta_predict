"""Tra cứu kết quả dự đoán ETA trên tập kiểm thử NYC Taxi."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "reports" / "predictions.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
MODELS = {
    "linear_regression": "Hồi quy tuyến tính",
    "random_forest": "Random Forest",
    "hgb": "Hist Gradient Boosting",
}


@st.cache_resource
def load_trips(predictions_version, test_version):
    # Thời điểm cập nhật của hai tệp là khóa cache khi chạy lại phần huấn luyện.
    predictions = pd.read_csv(
        PREDICTIONS, usecols=["id", "pickup_datetime", "trip_duration", "is_rush_hour", *MODELS],
        index_col="id", parse_dates=["pickup_datetime"],
    )
    distances = pd.read_csv(TEST, usecols=["id", "haversine_km", "manhattan_km"], index_col="id")
    trips = predictions.join(distances, validate="one_to_one")
    if trips[["haversine_km", "manhattan_km"]].isna().any().any():
        raise ValueError("Thiếu khoảng cách của chuyến kiểm thử. Kiểm tra hai tệp có cùng lần chạy.")
    return trips


def duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes} phút {seconds:02d} giây"


def arrival(start, seconds):
    end = start + pd.Timedelta(seconds=seconds)
    return end.strftime("%H:%M:%S" if end.date() == start.date() else "%d/%m/%Y %H:%M:%S")


def main():
    st.set_page_config(page_title="Tra cứu ETA · NYC Taxi", layout="wide")
    st.title("Tra cứu thời gian đến")
    st.write("Chọn một chuyến trong tập kiểm thử NYC Taxi để đối chiếu dự đoán của ba mô hình với thời điểm đến thực tế.")

    with st.form("lookup"):
        query = st.text_input("Mã chuyến kiểm thử", value="id1459588")
        st.form_submit_button("Xem kết quả", type="primary")
    trip_id = query.strip()
    if not trip_id:
        st.info("Nhập mã chuyến để tra cứu.")
        return
    try:
        trips = load_trips(PREDICTIONS.stat().st_mtime_ns, TEST.stat().st_mtime_ns)
    except FileNotFoundError as exc:
        st.error(f"Thiếu tệp {Path(exc.filename).name}. Chạy python -m src.train để tạo dữ liệu kiểm thử.")
        return
    except ValueError as exc:
        st.error(f"Dữ liệu tra cứu chưa hợp lệ. {exc}")
        return
    if trip_id not in trips.index:
        st.warning(f"Không tìm thấy mã {trip_id} trong tập kiểm thử.")
        return

    trip = trips.loc[trip_id]
    start = trip["pickup_datetime"]
    actual = int(trip["trip_duration"])
    st.subheader(f"Chuyến {trip_id}")
    st.write(f"Xuất phát lúc {start:%H:%M:%S}, ngày {start:%d/%m/%Y} (giờ New York).")
    distance, grid = st.columns(2)
    distance.metric("Haversine (đường chim bay)", f"{trip['haversine_km']:.3f} km".replace(".", ","))
    grid.metric("Manhattan (theo hai trục)", f"{trip['manhattan_km']:.3f} km".replace(".", ","),
                help="Cộng đoạn đổi vĩ độ và đoạn đổi kinh độ để xấp xỉ quãng đường theo lưới phố.")
    traffic, observed, destination = st.columns(3)
    traffic.metric("Khung giờ", "Cao điểm" if trip["is_rush_hour"] else "Ngoài cao điểm")
    observed.metric("Thời lượng thực tế", duration(actual))
    destination.metric("Giờ đến thực tế", arrival(start, actual))

    st.subheader("Đối chiếu dự đoán")
    rows = []
    for column, model_name in MODELS.items():
        seconds = round(trip[column])
        difference = seconds - actual
        comparison = "Đúng giờ"
        if difference:
            direction = "Muộn" if difference > 0 else "Sớm"
            comparison = f"{direction} {duration(abs(difference))}"
        rows.append({
            "Mô hình": model_name, "Thời lượng dự đoán": duration(seconds),
            "Giờ đến dự kiến (ETA)": arrival(start, seconds), "So với thực tế": comparison,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Thời lượng hiển thị được làm tròn đến giây. ETA bằng giờ xuất phát cộng thời lượng dự đoán.")


if __name__ == "__main__":
    main()
