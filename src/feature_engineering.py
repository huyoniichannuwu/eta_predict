import numpy as np

from .config import RUSH_WEEKDAYS, RUSH_HOURS

EARTH_RADIUS_KM = 6371          #ban kinh trung binh trai dat

FEATURES = [                    #gom 7 dac trung: khoang cach duong chim bay, manhattan, gio, ngay, gio_sin, gio_cos, caodiem(true/false)
    "haversine_km", "manhattan_km",
    "hour", "dayofweek", "hour_sin", "hour_cos",
    "is_rush_hour",
]


def haversine(lat1, lon1, lat2, lon2):
    """Khoảng cách đường chim bay (km) giữa hai điểm, đầu vào là độ."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_features(df):
    """Trả về bản sao của df có thêm các cột đặc trưng (tính riêng từng dòng)."""
    df = df.copy()
    plat, plon = df["pickup_latitude"], df["pickup_longitude"]
    dlat, dlon = df["dropoff_latitude"], df["dropoff_longitude"]

    df["haversine_km"] = haversine(plat, plon, dlat, dlon)
    # Manhattan: đi dọc theo vĩ độ rồi dọc theo kinh độ
    df["manhattan_km"] = haversine(plat, plon, dlat, plon) + haversine(dlat, plon, dlat, dlon)

    df["hour"] = df["pickup_datetime"].dt.hour
    
    #df["minute"] = df["pickup_datetime"].dt.minute              #them minute cho mục update hoursin
    # hour_with_minutes = df["hour"] + df["minute"] / 60
    # angle = 2 * np.pi * hour_with_minutes / 24

    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    
    
    angle = 2 * np.pi * df["hour"] / 24

    df["hour_sin"] = np.sin(angle)
    df["hour_cos"] = np.cos(angle)

    df["is_rush_hour"] = (
        df["dayofweek"].isin(RUSH_WEEKDAYS) & df["hour"].isin(RUSH_HOURS)
    ).astype(int)
    return df
