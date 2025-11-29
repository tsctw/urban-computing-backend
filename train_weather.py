# train_weather_fused.py
# 使用融合 + 校正後 Pressure_corrected 當輸入
# 預測未來 weather_main 分類序列

import io
import json
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

# 你自己的 fusion + calibration
from fusion import data_fusion, calibration

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 24
FUTURE_6_STEPS = 3
FUTURE_12_STEPS = 6
FUTURE_24_STEPS = 12

MODEL_OUTPUT = "model_weather.keras"
SCALER_OUTPUT = "scaler_weather.pkl"
LABEL_MAP_OUTPUT = "weather_label_map.json"


# -----------------------------
# 1. 取得融合 + 校正後資料
# -----------------------------
def load_fused_and_calibrated_data():
    print("🔄 Running data fusion...")
    aligned = data_fusion()

    print("🔧 Running calibration...")
    aligned = calibration(aligned)

    # 只保留 Pressure_corrected + 氣象特徵
    feature_cols = [
        "Pressure_corrected",
        "temperature",
        "humidity",
        "dew_point",
        "clouds",
        "wind_speed",
        "wind_deg",
    ]

    for c in feature_cols:
        if c not in aligned.columns:
            raise ValueError(f"Missing required feature: {c}")

    # X features
    features = aligned[feature_cols].values.astype("float32")

    # y labels
    weather_series = aligned["weather_main"].astype(str)
    labels, uniques = pd.factorize(weather_series)

    print("\n🌤 Weather classes found:", list(uniques))

    num_classes = len(uniques)

    id_to_label = {int(i): str(label) for i, label in enumerate(uniques)}

    # Scale X
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels, num_classes, scaler, id_to_label


# -----------------------------
# 2. 建立序列資料
# -----------------------------
def build_sequences(features, labels, num_classes):
    X = []
    y6_seq, y12_seq, y24_seq = [], [], []

    max_index = len(features) - (SEQ_LEN + FUTURE_24_STEPS)
    if max_index <= 0:
        raise ValueError("Not enough data to build sequences")

    for i in range(max_index):
        seq_x = features[i : i + SEQ_LEN]

        seq_6  = labels[i + SEQ_LEN : i + SEQ_LEN + FUTURE_6_STEPS]
        seq_12 = labels[i + SEQ_LEN : i + SEQ_LEN + FUTURE_12_STEPS]
        seq_24 = labels[i + SEQ_LEN : i + SEQ_LEN + FUTURE_24_STEPS]

        if len(seq_6) != FUTURE_6_STEPS: continue
        if len(seq_12) != FUTURE_12_STEPS: continue
        if len(seq_24) != FUTURE_24_STEPS: continue

        X.append(seq_x)
        y6_seq.append(tf.keras.utils.to_categorical(seq_6,  num_classes))
        y12_seq.append(tf.keras.utils.to_categorical(seq_12, num_classes))
        y24_seq.append(tf.keras.utils.to_categorical(seq_24, num_classes))

    X = np.array(X)
    y6_seq = np.array(y6_seq)
    y12_seq = np.array(y12_seq)
    y24_seq = np.array(y24_seq)

    print("\n📦 Sequence built:")
    print("  X:", X.shape)
    print("  y6:", y6_seq.shape)
    print("  y12:", y12_seq.shape)
    print("  y24:", y24_seq.shape)

    return X, [y6_seq, y12_seq, y24_seq]


# -----------------------------
# 3. GRU multi-step weather classifier
# -----------------------------
def build_weather_multistep_model(timesteps, feature_dim, num_classes):
    inputs = layers.Input(shape=(timesteps, feature_dim))

    x = layers.GRU(128, return_sequences=False)(inputs)
    x = layers.Dense(128, activation="relu")(x)

    w6  = layers.Dense(FUTURE_6_STEPS  * num_classes, activation="softmax")(x)
    w12 = layers.Dense(FUTURE_12_STEPS * num_classes, activation="softmax")(x)
    w24 = layers.Dense(FUTURE_24_STEPS * num_classes, activation="softmax")(x)

    w6  = layers.Reshape((FUTURE_6_STEPS,  num_classes), name="w6")(w6)
    w12 = layers.Reshape((FUTURE_12_STEPS, num_classes), name="w12")(w12)
    w24 = layers.Reshape((FUTURE_24_STEPS, num_classes), name="w24")(w24)

    model = models.Model(inputs, [w6, w12, w24])

    model.compile(
    optimizer="adam",
    loss={
        "w6": "categorical_crossentropy",
        "w12": "categorical_crossentropy",
        "w24": "categorical_crossentropy"
    },
    metrics={
        "w6": ["accuracy"],
        "w12": ["accuracy"],
        "w24": ["accuracy"]
    }
)

    return model


# -----------------------------
# MAIN TRAINING
# -----------------------------
if __name__ == "__main__":
    features_scaled, labels, num_classes, scaler, id_to_label = load_fused_and_calibrated_data()

    X, y_list = build_sequences(features_scaled, labels, num_classes)

    model = build_weather_multistep_model(SEQ_LEN, X.shape[-1], num_classes)
    model.summary()

    print("\n🚀 Training multi-step weather model...\n")

    history = model.fit(
        X,
        {"w6": y_list[0], "w12": y_list[1], "w24": y_list[2]},
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    model.save(MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    with open(LABEL_MAP_OUTPUT, "w") as f:
        json.dump(id_to_label, f, indent=2, ensure_ascii=False)

    print("\n🎉 Saved:", MODEL_OUTPUT)
    print("🎉 Saved:", SCALER_OUTPUT)
    print("🎉 Saved:", LABEL_MAP_OUTPUT)
