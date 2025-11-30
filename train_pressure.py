#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

from fusion import calibration, data_fusion, merge_data


# ================================
# 1. Load & Prepare Data
# ================================
def load_and_prepare_data():
    print("\n=== STEP 1: Running fusion → calibration → fusion2 ===")
    
    aligned = data_fusion()
    calibration_self = calibration(aligned)
    data = merge_data(calibration_self)

    print("✅ Data fusion complete. Shape:", data.shape)

    # Mainly used calibrated self pressure data, if no data then use weather pressure data
    data["Pressure_train"] = data["Pressure_self_corrected"].fillna(data["pressure"])

    X_raw = data["Pressure_train"].values.astype("float32").reshape(-1, 1)
    y_raw = X_raw.copy()

    # 標準化
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw)

    print("📦 X_raw shape:", X_raw.shape)
    print("📦 X_scaled shape:", X_scaled.shape)
    
    return X_scaled, y_scaled, scaler_x, scaler_y



# ================================
# 2. Build Sequences
# ================================
def build_sequences(X, y, seq_len=24, f6=6, f12=12, f24=24):
    print("\n=== STEP 2: Building sequential dataset ===")

    X_seq = []
    y6_seq, y12_seq, y24_seq = [], [], []

    max_index = len(X) - (seq_len + f24)
    if max_index <= 0:
        raise ValueError("❌ Data too short for sequence training!")

    for i in range(max_index):
        X_seq.append(X[i:i+seq_len])
        
        y6_seq.append(y[i+seq_len : i+seq_len+f6])
        y12_seq.append(y[i+seq_len : i+seq_len+f12])
        y24_seq.append(y[i+seq_len : i+seq_len+f24])

    X_seq  = np.array(X_seq)
    y6_seq = np.array(y6_seq)
    y12_seq = np.array(y12_seq)
    y24_seq = np.array(y24_seq)

    print("📊 X_seq:", X_seq.shape)
    print("📊 y6_seq:", y6_seq.shape)
    print("📊 y12_seq:", y12_seq.shape)
    print("📊 y24_seq:", y24_seq.shape)

    return X_seq, [y6_seq, y12_seq, y24_seq]



# ================================
# 3. Build LSTM Encoder–Decoder Model
# ================================
def build_pressure_model(seq_len):
    print("\n=== STEP 3: Building LSTM Encoder–Decoder model ===")

    inputs = layers.Input(shape=(seq_len, 1))

    # Encoder
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(64, return_sequences=False)(x)

    # Shared embedding for all horizons
    latent = layers.Dense(64, activation="relu")(x)

    # ----------------------------
    # Decoder for 6h
    # ----------------------------
    d6 = layers.RepeatVector(6)(latent)
    d6 = layers.LSTM(64, return_sequences=True)(d6)
    p6 = layers.TimeDistributed(layers.Dense(1), name="p6")(d6)

    # ----------------------------
    # Decoder for 12h
    # ----------------------------
    d12 = layers.RepeatVector(12)(latent)
    d12 = layers.LSTM(64, return_sequences=True)(d12)
    p12 = layers.TimeDistributed(layers.Dense(1), name="p12")(d12)

    # ----------------------------
    # Decoder for 24h
    # ----------------------------
    d24 = layers.RepeatVector(24)(latent)
    d24 = layers.LSTM(64, return_sequences=True)(d24)
    p24 = layers.TimeDistributed(layers.Dense(1), name="p24")(d24)

    model = models.Model(inputs, [p6, p12, p24])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={"p6": "mse", "p12": "mse", "p24": "mse"},
        loss_weights={"p6": 0.5, "p12": 0.3, "p24": 0.2},
        metrics={"p6": ["mse"], "p12": ["mse"], "p24": ["mse"]}
    )

    model.summary()
    return model



# ================================
# 4. MAIN TRAINING PIPELINE
# ================================
if __name__ == "__main__":
    print("\n🚀 TRAINING: Pressure Forecast Model (6h / 12h / 24h)\n")

    # Load + scale
    X_scaled, y_scaled, scaler_x, scaler_y = load_and_prepare_data()

    # Build sequences
    X_seq, y_list = build_sequences(X_scaled, y_scaled)

    # Create model
    model = build_pressure_model(24)

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=4,
            factor=0.5
        )
    ]

    print("\n=== STEP 4: Training... ===")
    history = model.fit(
        X_seq,
        {"p6": y_list[0], "p12": y_list[1], "p24": y_list[2]},
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # ================================
    # Save model
    # ================================
    print("\n💾 Saving model + scalers...")
    model.save("model_pressure_single.keras")
    joblib.dump(scaler_x, "scaler_pressure_x.pkl")
    joblib.dump(scaler_y, "scaler_pressure_y.pkl")

    print("\n🎉 TRAINING COMPLETE!")
    print("📁 Saved: model_pressure_single.keras")
    print("📁 Saved: scaler_pressure_x.pkl")
    print("📁 Saved: scaler_pressure_y.pkl")
