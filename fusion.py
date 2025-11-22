from utils import read_gcs_csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

SELF_DATA_BUCKET_NAME = "urban-computing-self-data-processed"
SELF_DATA_FILE = "merged.csv"

WEATHER_DATA_BUCKET_NAME = "urban-computing-api-data-processed"
ALL_DATA_FILE = "all_data.csv"

self_df = read_gcs_csv(SELF_DATA_BUCKET_NAME, SELF_DATA_FILE)
weather_df = read_gcs_csv(WEATHER_DATA_BUCKET_NAME, ALL_DATA_FILE)

self_df["Time"] = pd.to_datetime(self_df["Time"], unit="s", utc=True).dt.tz_convert("Europe/Dublin")

weather_df["Time"] = pd.to_datetime(weather_df["Time"], unit="s", utc=True).dt.tz_convert("Europe/Dublin")

# 找出 weather 資料最早的 timestamp
start_time = weather_df["Time"].min()

# 過濾 self data（保留在 weather 有資料之後的資料）
self_df = self_df[self_df["Time"] >= start_time]

aligned = pd.merge_asof(
    self_df.sort_values("Time"),
    weather_df.sort_values("Time"),
    on="Time",
    direction="nearest",                # 取最近的時間點
    tolerance=pd.Timedelta("2h"),       # 最多差 2 小時（可調）
    suffixes=("_self", "_weather")
)

aligned["diff_before"] = aligned["Pressure_self"] - aligned["Pressure_weather"]


# bias = aligned["diff"].mean()
# print(bias)

# std = aligned["diff"].std()
# print(std)

# aligned.plot(x="Time", y=["Pressure_self", "Pressure_weather"], figsize=(10,5))
# plt.show()

# aligned.plot(x="Time", y="diff", kind="line", figsize=(10,4))
# plt.show()

# --------------------------------------------------------
# 3. Compute Pre-Calibration Error
# --------------------------------------------------------
aligned["diff_before"] = aligned["Pressure_self"] - aligned["Pressure_weather"]

mean_before = aligned["diff_before"].mean()
std_before = aligned["diff_before"].std()

print(f"Calibration BEFORE:")
print(f"  Mean difference (self - weather): {mean_before:.4f} hPa")
print(f"  Standard deviation: {std_before:.4f} hPa")

# --------------------------------------------------------
# 4. Train Linear Regression Calibration Model
# --------------------------------------------------------
model = LinearRegression()
model.fit(
    aligned[["Pressure_self"]],
    aligned["Pressure_weather"]
)

a = model.coef_[0]
b = model.intercept_

print("\nCalibration Model:")
print(f"  P_corrected = {a:.6f} * P_self + {b:.6f}")


# --------------------------------------------------------
# 5. Apply Calibration
# --------------------------------------------------------
aligned["Pressure_corrected"] = a * aligned["Pressure_self"] + b

# Compute corrected error
aligned["diff_after"] = aligned["Pressure_corrected"] - aligned["Pressure_weather"]

mean_after = aligned["diff_after"].mean()
std_after = aligned["diff_after"].std()

print(f"\nCalibration AFTER:")
print(f"  Mean difference: {mean_after:.4f} hPa")
print(f"  Standard deviation: {std_after:.4f} hPa")


# --------------------------------------------------------
# 6. Plot Comparison
# --------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(aligned["Time"], aligned["diff_before"], label="Before Calibration", alpha=0.7)
plt.plot(aligned["Time"], aligned["diff_after"], label="After Calibration", alpha=0.7)
plt.axhline(0, color="black", linestyle="--")
plt.title("Pressure Difference (Self vs Weather) Before / After Calibration")
plt.ylabel("Difference (hPa)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Scatter plot to visualize regression
plt.figure(figsize=(6, 6))
plt.scatter(aligned["Pressure_self"], aligned["Pressure_weather"], alpha=0.6, label="Raw Data")
plt.plot(aligned["Pressure_self"], aligned["Pressure_corrected"], color="red", label="Corrected Fit")
plt.xlabel("Self Pressure (hPa)")
plt.ylabel("Weather Pressure (hPa)")
plt.title("Calibration Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()