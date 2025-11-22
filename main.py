from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
from utils import read_gcs_csv, process_zip_file
import pandas as pd
import json


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