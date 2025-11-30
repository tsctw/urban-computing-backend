# train_weather_fused_lstm.py
# 使用融合 + 校正後 Pressure_corrected 當輸入
# 預測未來 weather_main 分類序列（LSTM 版本）

import io
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

from fusion import data_fusion, calibration, data_fusion_2

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 24
FUTURE_6 = 3
FUTURE_12 = 6
FUTURE_24 = 12

MODEL_OUTPUT = "model_weather.keras"
SCALER_OUTPUT = "scaler_weather.pkl"
LABEL_MAP_OUTPUT = "weather_label_map.json"


# -----------------------------
# 1. 融合 + 校正後資料
# -----------------------------
def load_fused_and_calibrated_data():
    print("🔄 Running data fusion...")
    aligned = data_fusion()

    print("🔧 Running calibration...")
    calibration_self = calibration(aligned)
    data = data_fusion_2(calibration_self)

    feature_cols = [
        "Pressure_self_corrected",
        "temperature",
        "humidity",
        "dew_point",
        "clouds",
        "wind_speed",
        "wind_deg",
    ]

    for c in feature_cols:
        if c not in data.columns:
            raise ValueError(f"Missing feature: {c}")

    features = data[feature_cols].values.astype("float32")

    weather_series = data["weather_main"].astype(str)
    labels, uniques = pd.factorize(weather_series)

    print("\n🌤 Weather classes:", list(uniques))

    num_classes = len(uniques)
    id_to_label = {int(i): str(label) for i, label in enumerate(uniques)}

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels, num_classes, scaler, id_to_label


# -----------------------------
# 2. 序列資料構建（分類 one-hot）
# -----------------------------
def build_sequences(features, labels, num_classes):
    X, y6, y12, y24 = [], [], [], []

    max_idx = len(features) - (SEQ_LEN + FUTURE_24)
    if max_idx <= 0:
        raise ValueError("Not enough data")

    for i in range(max_idx):
        X.append(features[i:i+SEQ_LEN])

        y6.append(tf.keras.utils.to_categorical(
            labels[i+SEQ_LEN : i+SEQ_LEN+FUTURE_6], num_classes))
        y12.append(tf.keras.utils.to_categorical(
            labels[i+SEQ_LEN : i+SEQ_LEN+FUTURE_12], num_classes))
        y24.append(tf.keras.utils.to_categorical(
            labels[i+SEQ_LEN : i+SEQ_LEN+FUTURE_24], num_classes))

    X = np.array(X)
    y6 = np.array(y6)
    y12 = np.array(y12)
    y24 = np.array(y24)

    print("\n📦 Sequence shapes:")
    print(" X:", X.shape)
    print(" y6:", y6.shape)
    print(" y12:", y12.shape)
    print(" y24:", y24.shape)

    return X, [y6, y12, y24]


# -----------------------------
# 3. LSTM Encoder–Decoder 多步分類模型
# -----------------------------
def build_weather_model(feature_dim, num_classes):
    inputs = layers.Input(shape=(SEQ_LEN, feature_dim))

    # Encoder
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(64, return_sequences=False)(x)

    # Shared embedding
    latent = layers.Dense(64, activation="relu")(x)

    # ------------- Decoder 6h -------------
    d6 = layers.RepeatVector(FUTURE_6)(latent)
    d6 = layers.LSTM(64, return_sequences=True)(d6)
    out6 = layers.TimeDistributed(
        layers.Dense(num_classes, activation="softmax"),
        name="w6"
    )(d6)

    # ------------- Decoder 12h -------------
    d12 = layers.RepeatVector(FUTURE_12)(latent)
    d12 = layers.LSTM(64, return_sequences=True)(d12)
    out12 = layers.TimeDistributed(
        layers.Dense(num_classes, activation="softmax"),
        name="w12"
    )(d12)

    # ------------- Decoder 24h -------------
    d24 = layers.RepeatVector(FUTURE_24)(latent)
    d24 = layers.LSTM(64, return_sequences=True)(d24)
    out24 = layers.TimeDistributed(
        layers.Dense(num_classes, activation="softmax"),
        name="w24"
    )(d24)

    model = models.Model(inputs, [out6, out12, out24])

    model.compile(
        optimizer="adam",
        loss={
            "w6":  "categorical_crossentropy",
            "w12": "categorical_crossentropy",
            "w24": "categorical_crossentropy",
        },
        loss_weights={"w6": 0.5, "w12": 0.3, "w24": 0.2},
        metrics={"w6": "accuracy", "w12": "accuracy", "w24": "accuracy"},
    )

    return model


# -----------------------------
# MAIN TRAINING
# -----------------------------
if __name__ == "__main__":
    features_scaled, labels, num_classes, scaler, id_to_label = load_fused_and_calibrated_data()

    X, y_list = build_sequences(features_scaled, labels, num_classes)

    model = build_weather_model(X.shape[-1], num_classes)
    model.summary()

    print("\n🚀 Training LSTM multi-step weather model...\n")

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    history = model.fit(
        X,
        {"w6": y_list[0], "w12": y_list[1], "w24": y_list[2]},
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        callbacks=cb,
        verbose=1,
    )

    model.save(MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    with open(LABEL_MAP_OUTPUT, "w") as f:
        json.dump(id_to_label, f, indent=2, ensure_ascii=False)

    print("\n🎉 Saved:", MODEL_OUTPUT)
    print("🎉 Saved:", SCALER_OUTPUT)
    print("🎉 Saved:", LABEL_MAP_OUTPUT)
