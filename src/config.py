from pathlib import Path

# Thư mục gốc dự án = cha của thư mục src
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT_DIR / "data" / "raw" / "trips_raw.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RANDOM_SEED = 42

# Ngưỡng lọc trip_duration (giây): loại ~0.73% chuyến lỗi/ngoại lai - này đã tìm trong notebook (cai bat thuong cua trip_duration)
MIN_DURATION = 60
MAX_DURATION = 10800

# Giờ cao điểm
RUSH_WEEKDAYS = [0, 1, 2, 3, 4]   # thứ 2 đến thứ 6 (Monday = 0, Friday = 4)
RUSH_HOURS = [6, 7, 8, 9 , 16, 17, 18, 19]   #giai thich: 9: 9h00 - 9h59, nguồn dựa vào references/traffic_reference