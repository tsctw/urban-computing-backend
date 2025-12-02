from utils import read_gcs_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


SEQ_LEN = 24

def data_fusion():
    self_df = pd.read_csv("merged.csv")
    weather_df = pd.read_csv("all_data.csv")

    self_df["Time"] = pd.to_datetime(self_df["Time"], unit="s", utc=True).dt.tz_convert("Europe/Dublin")

    weather_df["Time"] = pd.to_datetime(weather_df["timestamp"], unit="s", utc=True).dt.tz_convert("Europe/Dublin")

    # get the earliest weather timestamp
    start_time = weather_df["Time"].min()

    # filter self data after weather data is available
    self_df = self_df[self_df["Time"] >= start_time]

    self_df.to_csv("self_df__output.csv", index=False)

    aligned = pd.merge_asof(
        self_df.sort_values("Time"),
        weather_df.sort_values("Time"),
        on="Time",
        direction="nearest",                # get nearest time point
        tolerance=pd.Timedelta("2h"),       # max difference is 2 hr
        suffixes=("_self", "_weather")
    )

    aligned = aligned.rename(columns={
        "Pressure": "Pressure_self",
        "pressure": "Pressure_weather"
    })

    aligned.to_csv("fusion_output.csv", index=False)
    
    aligned["diff_before"] = aligned["Pressure_self"] - aligned["Pressure_weather"]

    return aligned


def draw_fusion(aligned):

    bias = aligned["diff_before"].mean()
    print(bias)

    std = aligned["diff_before"].std()
    print(std)

    aligned.plot(x="Time", y=["Pressure_self", "Pressure_weather"], figsize=(10,5))
    plt.show()

    aligned.plot(x="Time", y="diff_before", kind="line", figsize=(10,4))
    plt.show()

# --------------------------------------------------------
# 3. Compute Pre-Calibration Error
# --------------------------------------------------------

def calibration(aligned, n=10):

    mean_before = aligned["diff_before"].mean()
    std_before = aligned["diff_before"].std()

    print(f"Calibration BEFORE:")
    print(f"  Mean difference (self - weather): {mean_before:.4f} hPa")
    print(f"  Standard deviation: {std_before:.4f} hPa")

    # --------------------------------------------------------
    # Train Linear Regression Calibration Model
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
    # Apply Calibration
    # --------------------------------------------------------
    aligned["Pressure_corrected"] = a * aligned["Pressure_self"] + b

    # Compute corrected error
    aligned["diff_after"] = aligned["Pressure_corrected"] - aligned["Pressure_weather"]


    aligned.to_csv("calibration_output.csv", index=False)

    mean_after = aligned["diff_after"].mean()
    std_after = aligned["diff_after"].std()

    print(f"\nCalibration AFTER:")
    print(f"  Mean difference: {mean_after:.4f} hPa")
    print(f"  Standard deviation: {std_after:.4f} hPa")
    return aligned


# --------------------------------------------------------
# 6. Plot Comparison
# --------------------------------------------------------
def draw_calibration(aligned):
    plt.figure(figsize=(12, 5))
    plt.plot(aligned["Time"], aligned["diff_before"], label="Before Calibration", alpha=0.7)
    plt.plot(aligned["Time"], aligned["diff_after"], label="After Calibration", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Pressure Difference (Self/Calibrated Self vs Weather) Before / After Calibration")
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

def compress_by_5min_gap(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Time","Pressure_self", "Pressure_weather", "Pressure_corrected"]].copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")

    if len(df) == 0:
        return df

    groups = []
    current_start_time = df.iloc[0]["Time"]
    current_raw_values = [df.iloc[0]["Pressure_self"]]
    current_weather_pressure_value = [df.iloc[0]["Pressure_weather"]]
    current_values = [df.iloc[0]["Pressure_corrected"]]
    prev_time = df.iloc[0]["Time"]

    for i in range(1, len(df)):
        t = df.iloc[i]["Time"]
        s = df.iloc[i]["Pressure_self"]
        w = df.iloc[i]["Pressure_weather"]
        p = df.iloc[i]["Pressure_corrected"]

        if t - prev_time < pd.Timedelta(minutes=5):
            current_raw_values.append(s)
            current_values.append(p)
        else:
            groups.append({
                "Time": current_start_time,
                "Pressure_self": sum(current_raw_values) / len(current_raw_values),
                "Pressure_weather": current_weather_pressure_value,
                "Pressure_corrected": sum(current_values) / len(current_values)
            })
            current_start_time = t
            current_raw_values = [s]
            current_weather_pressure_value = w
            current_values = [p]

        prev_time = t

    groups.append({
        "Time": current_start_time,
        "Pressure_self": sum(current_raw_values) / len(current_raw_values),
        "Pressure_weather": current_weather_pressure_value,
        "Pressure_corrected": sum(current_values) / len(current_values)
    })

    out_df = pd.DataFrame(groups).sort_values("Time").reset_index(drop=True)
    return out_df

def merge_data(calibration_self):
    # -----------------------------------
    # 1. Load data
    # -----------------------------------
    weather_df = pd.read_csv("all_data.csv")

    # Convert timestamp
    weather_df["Time"] = pd.to_datetime(weather_df["timestamp"], unit="s", utc=True).dt.tz_convert("Europe/Dublin")

    calibration_self = compress_by_5min_gap(calibration_self)
    calibration_self.to_csv("compress_self_output.csv", index=False)

    # three different pressure
    self_times_ns = calibration_self["Time"].astype("int64").to_numpy()

    self_pressures_corrected = calibration_self["Pressure_corrected"].to_numpy()
    self_pressures_raw = calibration_self["Pressure_self"].to_numpy()

    ONE_HOUR_NS = 3600 * 1_000_000_000

    results = []

    for _, w in weather_df.iterrows():
        t_weather = w["Time"]
        t_weather_ns = t_weather.value  # Timestamp → ns

        # cal time difference
        diffs = np.abs(self_times_ns - t_weather_ns)

        # in 1 hour
        within_range = diffs <= ONE_HOUR_NS

        if within_range.any():
            idx = diffs.argmin()

            matched_corrected = self_pressures_corrected[idx]
            matched_self = self_pressures_raw[idx]

        else:
            matched_corrected = None
            matched_self = None
            matched_weather = None

        row_dict = w.to_dict()

        row_dict["Pressure_self_corrected"] = (
            matched_corrected if matched_corrected is not None else row_dict["pressure"]
        )
        row_dict["Pressure_self"] = (
            matched_self if matched_self is not None else None
        )

        results.append(row_dict)
    
    merged_df = pd.DataFrame(results).sort_values("Time")

    merged_df.to_csv("merge_output.csv", index=False)

    return merged_df

def get_latest_pressure_sequence(scaler_x):
    # 1. fusion
    aligned = data_fusion()

    # 2. calibration
    aligned = calibration(aligned)

    # 3. ensure enough data
    if len(aligned) < SEQ_LEN:
        raise ValueError("Not enough data to form a 24-step sequence.")

    # 4. take the last 24 Pressure_corrected
    seq = aligned["Pressure_corrected"].values[-SEQ_LEN:].astype("float32")

    # shape (24, 1)
    seq = seq.reshape(-1, 1)

    # 5. scale X using scaler_x
    seq_scaled = scaler_x.transform(seq)

    # model expects shape: (1, 24, 1)
    seq_scaled = seq_scaled.reshape(1, SEQ_LEN, 1)

    return seq_scaled

SEQ_LEN = 24

def get_latest_weather_sequence(scaler_weather):

    FEATURE_COLS = [
    "Pressure_corrected",
    "temperature",
    "humidity",
    "dew_point",
    "clouds",
    "wind_speed",
    "wind_deg",
    ]

    aligned = data_fusion()
    aligned = calibration(aligned)

    if len(aligned) < SEQ_LEN:
        raise ValueError("Not enough data for 24-step sequence.")

    # get features in last 24 row
    seq = aligned[FEATURE_COLS].values[-SEQ_LEN:].astype("float32")

    seq_scaled = scaler_weather.transform(seq)

    # GRU expects shape: (1, 24, feature_dim)
    return seq_scaled.reshape(1, SEQ_LEN, len(FEATURE_COLS))


if __name__ == "__main__":
    aligned = data_fusion()
    # draw_fusion(aligned)
    calibration_self = calibration(aligned)
    draw_calibration(aligned)
    # merge_data(calibration_self)