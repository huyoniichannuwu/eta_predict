import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAW_PATH, PROCESSED_DIR, RANDOM_SEED, MIN_DURATION, MAX_DURATION

N_SAMPLE = None  # Dùng toàn bộ dữ liệu đã làm sạch
TEST_SIZE = 0.2


def clean(df):
    """Lọc chuyến lỗi/ngoại lai và đổi pickup_datetime sang kiểu ngày giờ."""
    df = df[df["trip_duration"].between(MIN_DURATION, MAX_DURATION)].copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df


def sample_and_split(df, split_mode="random"):
    """Lấy mẫu N dòng rồi chia 80/20 theo 'random' hoặc 'time'."""
    if N_SAMPLE is not None:
        df = df.sample(n=N_SAMPLE, random_state=RANDOM_SEED)

    if split_mode == "random":
        return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    if split_mode == "time":
        df = df.sort_values("pickup_datetime")
        cut = int(len(df) * (1 - TEST_SIZE))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    raise ValueError("split_mode phải là 'random' hoặc 'time'")


def load_split(split_mode="random"):
    """Đọc CSV -> làm sạch -> lấy mẫu -> chia. Trả về (train, test)."""
    df = pd.read_csv(RAW_PATH)
    df = clean(df)
    return sample_and_split(df, split_mode)


def save_clean_data():
    """Làm sạch toàn bộ dữ liệu gốc và lưu vào data/processed/trips_clean.csv."""
    df = pd.read_csv(RAW_PATH)
    cleaned = clean(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "trips_clean.csv"
    cleaned.to_csv(output_path, index=False)
    print(f"Dữ liệu gốc: {len(df):,} chuyến")
    print(f"Sau làm sạch: {len(cleaned):,} chuyến")
    print(f"Đã lưu: {output_path}")


if __name__ == "__main__":
    save_clean_data()