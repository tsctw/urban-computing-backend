from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from utils import read_gcs_csv, process_zip_file

app = FastAPI(title="Urban Computing API")

# 定義資料模型（可選）
class SensorData(BaseModel):
    user_id: str
    pressure: float
    temperature: float
    timestamp: str

# 根路徑
@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running!"}

SELF_DATA_BUCKET_NAME = "urban-computing-self-data-processed"
SELF_DATA_FILE = "merged.csv"

@app.get("/selfdata")
def get_self_data():
    """
    使用共用函式讀取 merged.csv，回傳 JSON 格式。
    """
    df = read_gcs_csv(SELF_DATA_BUCKET_NAME, SELF_DATA_FILE)
    records = df.to_dict(orient="records")
    return {"status": "success", "records": records}

WEATHER_DATA_BUCKET_NAME = "urban-computing-api-data-processed"
ALL_DATA_FILE = "all_data.csv"

@app.get("/openweatherdata")
def get_open_weather_data():
    df = read_gcs_csv(WEATHER_DATA_BUCKET_NAME, ALL_DATA_FILE)
    records = df.to_dict(orient="records")
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