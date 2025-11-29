from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
from utils import read_gcs_csv, process_zip_file
from fusion import get_latest_pressure_sequence, get_latest_weather_sequence
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import joblib


app = FastAPI(title="Urban Computing API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://100.65.146.94:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SensorData(BaseModel):
    user_id: str
    pressure: float
    temperature: float
    timestamp: str

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running!"}

SELF_DATA_BUCKET_NAME = "urban-computing-self-data-processed"
SELF_DATA_FILE = "merged.csv"

@app.get("/selfdata")
def get_self_data():
    df = read_gcs_csv(SELF_DATA_BUCKET_NAME, SELF_DATA_FILE)
    records = df.to_dict(orient="records")
    return {"status": "success", "records": records}

WEATHER_DATA_BUCKET_NAME = "urban-computing-api-data-processed"
ALL_DATA_FILE = "all_data.csv"

@app.get("/openweatherdata")
def get_open_weather_data():
    df = read_gcs_csv(WEATHER_DATA_BUCKET_NAME, ALL_DATA_FILE)

    # 🛠 修正 NaN / Inf → 轉成 None，避免 JSON encode 崩潰
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.fillna(pd.NA)

    # 用 pandas.to_json 保證正確序列化
    records = json.loads(df.to_json(orient="records"))

    return {"status": "success", "records": records}


# Upload single file
@app.post("/upload")
async def upload_zip(file: UploadFile = File(...)):
    storage_client = storage.Client()
    try:
        filename = process_zip_file(file, storage_client)
        return {"status": "success", "file": filename}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Upload batch files
@app.post("/upload_batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    storage_client = storage.Client()
    success_files = []
    failed_files = []

    for file in files:
        try:
            filename = process_zip_file(file, storage_client)
            success_files.append(filename)
        except Exception as e:
            print(f"❌ Failed {file.filename}: {e}")
            failed_files.append(file.filename)

    print("🎯 Batch upload finished.")
    return {
        "status": "completed",
        "success_files": success_files,
        "failed_files": failed_files,
        "success_count": len(success_files),
        "failed_count": len(failed_files),
    }

# -------------------------------------------------------------
# Load model + scalers ONCE (avoid loading at each request)
# -------------------------------------------------------------
model = None
scaler_x = None
scaler_y = None

MODEL_PATH = "model_pressure_single.keras"
SCALER_X_PATH = "scaler_pressure_x.pkl"
SCALER_Y_PATH = "scaler_pressure_y.pkl"

model = tf.keras.models.load_model("model_pressure_single.keras")
scaler_x = joblib.load("scaler_pressure_x.pkl")
scaler_y = joblib.load("scaler_pressure_y.pkl")


@app.get("/prediction_pressure")
def prediction_pressure():
    try:
        seq_input = get_latest_pressure_sequence(scaler_x)

        # model output: [p6, p12, p24] (all scaled)
        p6_scaled, p12_scaled, p24_scaled = model.predict(seq_input)

        # inverse transform
        p6 = scaler_y.inverse_transform(p6_scaled.reshape(-1, 1)).flatten().tolist()
        p12 = scaler_y.inverse_transform(p12_scaled.reshape(-1, 1)).flatten().tolist()
        p24 = scaler_y.inverse_transform(p24_scaled.reshape(-1, 1)).flatten().tolist()

        # map output to user request
        result = {
            "p6": p6,
            "p12": p12,
            "p24": p24,
        }

        return result

    except Exception as e:
        return {"error": str(e)}


model_weather = tf.keras.models.load_model("model_weather.keras")
scaler_weather = joblib.load("scaler_weather.pkl")

with open("weather_label_map.json", "r") as f:
    id_to_label = json.load(f)

@app.get("/prediction_weather")
def prediction_weather():

    try:
        seq_input = get_latest_weather_sequence(scaler_weather)

        # model returns 3 outputs: (w6, w12, w24)
        w6_s, w12_s, w24_s = model_weather.predict(seq_input)

        # softmax → label
        def decode_weather(softmax_arr):
            labels = []
            for step in softmax_arr:
                idx = int(np.argmax(step))
                labels.append(id_to_label[str(idx)])
            return labels

        w6  = decode_weather(w6_s[0])
        w12 = decode_weather(w12_s[0])
        w24 = decode_weather(w24_s[0])

        return {
            "weather": {
                "w6":  w6,   # 3 items
                "w12": w12,  # 6 items
                "w24": w24   # 12 items
            }
        }

    except Exception as e:
        return {"error": str(e)}
