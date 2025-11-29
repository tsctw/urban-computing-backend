import io
import json
from fusion import calibration, data_fusion
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "urban-computing-project-476914"
BUCKET_NAME = "urban-computing-api-data-processed"
FILE_NAME = "all_data.csv"
LOCAL_CSV = "all_data.csv"

SEQ_LEN = 24
FUTURE_6_STEPS = 3
FUTURE_12_STEPS = 6
FUTURE_24_STEPS = 12

MODEL_OUTPUT = "model_pressure_single.keras"
SCALER_X_OUTPUT = "scaler_pressure_x.pkl"
SCALER_Y_OUTPUT = "scaler_pressure_y.pkl"


# -----------------------------
# 1. 從 GCS 下載 CSV
# -----------------------------
def download_csv_from_gcs():
    print("📥 Downloading CSV from Google Cloud Storage...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_NAME)

    if not blob.exists():
        raise FileNotFoundError(f"{FILE_NAME} not found in GCS!")

    content = blob.download_as_text()
    with open(LOCAL_CSV, "w") as f:
        f.write(content)

    print("✅ Downloaded:", LOCAL_CSV)


# -----------------------------
# 2. 載入資料（單一特徵：pressure）
# -----------------------------
def load_and_prepare_data():
    # Step 1: run fusion
    aligned = data_fusion()

    # Step 2: run calibration
    aligned = calibration(aligned)

    # 必須確認校正後的欄位存在
    if "Pressure_corrected" not in aligned.columns:
        raise ValueError("Pressure_corrected column not found! Calibration failed.")

    # X,y = 校正後壓力
    X_raw = aligned["Pressure_corrected"].values.astype("float32").reshape(-1, 1)
    y_raw = X_raw.copy()

    # 標準化
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw).flatten()

    print("⬆ X shape:", X_scaled.shape)
    print("⬆ y shape:", y_scaled.shape)

    return X_scaled, y_scaled, scaler_x, scaler_y



# -----------------------------
# 3. 建立序列資料
# -----------------------------
def build_sequences(X, y):
    X_seq = []
    y6_seq, y12_seq, y24_seq = [], [], []

    max_index = len(X) - (SEQ_LEN + FUTURE_24_STEPS)
    if max_index <= 0:
        raise ValueError("Insufficient data")

    for i in range(max_index):
        seq_x = X[i : i + SEQ_LEN]  # shape: (24, 1)

        y6  = y[i + SEQ_LEN : i + SEQ_LEN + FUTURE_6_STEPS]
        y12 = y[i + SEQ_LEN : i + SEQ_LEN + FUTURE_12_STEPS]
        y24 = y[i + SEQ_LEN : i + SEQ_LEN + FUTURE_24_STEPS]

        if len(y6)==3 and len(y12)==6 and len(y24)==12:
            X_seq.append(seq_x)
            y6_seq.append(y6)
            y12_seq.append(y12)
            y24_seq.append(y24)

    X_seq = np.array(X_seq)
    y6_seq = np.array(y6_seq)
    y12_seq = np.array(y12_seq)
    y24_seq = np.array(y24_seq)

    print("📦 Sequence shapes:")
    print("X:", X_seq.shape)
    print("y6:", y6_seq.shape)
    print("y12:", y12_seq.shape)
    print("y24:", y24_seq.shape)

    return X_seq, [y6_seq, y12_seq, y24_seq]


# -----------------------------
# 4. GRU 模型（單特徵）
# -----------------------------
def build_pressure_model(timesteps, feature_dim):
    inputs = layers.Input(shape=(timesteps, feature_dim))

    x = layers.GRU(96, return_sequences=False)(inputs)
    x = layers.Dense(64, activation="relu")(x)

    p6  = layers.Dense(FUTURE_6_STEPS, activation="linear", name="p6")(x)
    p12 = layers.Dense(FUTURE_12_STEPS, activation="linear", name="p12")(x)
    p24 = layers.Dense(FUTURE_24_STEPS, activation="linear", name="p24")(x)

    model = models.Model(inputs, [p6, p12, p24])

    model.compile(
        optimizer="adam",
        loss={"p6": "mse", "p12": "mse", "p24": "mse"},
        metrics={"p6": ["mse"], "p12": ["mse"], "p24": ["mse"]}
    )

    return model


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    download_csv_from_gcs()

    X_scaled, y_scaled, scaler_x, scaler_y = load_and_prepare_data()
    X, y_list = build_sequences(X_scaled, y_scaled)

    model = build_pressure_model(SEQ_LEN, X.shape[-1])
    model.summary()

    print("🚀 Training pressure model (single feature)...")
    history = model.fit(
        X,
        {"p6": y_list[0], "p12": y_list[1], "p24": y_list[2]},
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    # 儲存
    model.save(MODEL_OUTPUT)
    joblib.dump(scaler_x, SCALER_X_OUTPUT)
    joblib.dump(scaler_y, SCALER_Y_OUTPUT)

    print("🎉 Saved model + scalers!")
