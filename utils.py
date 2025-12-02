from fastapi import UploadFile, HTTPException
from google.cloud import storage
import pandas as pd
import zipfile
from io import BytesIO, StringIO

def read_gcs_csv(bucket_name: str, file_name: str):

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    data = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data))
    return df

BUCKET_NAME_SOURCE = "urban-computing-self-data-raw"
BUCKET_NAME_DEST = "urban-computing-self-data-processed"
OUTPUT_PATH = "merged.csv"

def process_zip_file(file: UploadFile, storage_client: storage.Client) -> str:

    # verify format
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail=f"{file.filename} is not a .zip file")

    # clear space and filename
    new_name = file.filename.lstrip().replace(" ", "_")
    print(f"📦 Processing ZIP: {new_name}")

    # init GCS bucket
    bucket_source = storage_client.bucket(BUCKET_NAME_SOURCE)
    bucket_dest = storage_client.bucket(BUCKET_NAME_DEST)
    blob = bucket_source.blob(new_name)

    # check repeat
    if blob.exists():
        print(f"⚠️ File already exists: {new_name}")
        raise HTTPException(status_code=409, detail=f"File '{new_name}' already exists in {BUCKET_NAME_SOURCE}")

    # zip to memory
    file.file.seek(0)
    zip_bytes = file.file.read()

    try:
        zf = zipfile.ZipFile(BytesIO(zip_bytes))
        nameList = zf.namelist()
        print(f"🔍 ZIP Content: {nameList}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {e}")

    # verify necessary data
    if "Raw Data.csv" not in nameList or "meta/time.csv" not in nameList:
        raise HTTPException(status_code=400, detail=f"{file.filename} missing required files (Raw Data.csv or meta/time.csv)")

    # read meta/time.csv
    try:
        df_meta = pd.read_csv(zf.open("meta/time.csv"))
        start_row = df_meta[df_meta["event"] == "START"].iloc[0]
        start_system_time = float(start_row["system time"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading meta/time.csv: {e}")

    # read Raw Data.csv
    try:
        df_raw = pd.read_csv(zf.open("Raw Data.csv"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading Raw Data.csv: {e}")

    # find time and pressure columns
    time_col = [c for c in df_raw.columns if "time" in c.lower()][0]
    pressure_col = [c for c in df_raw.columns if "press" in c.lower()][0]

    # process data
    df_raw["Time"] = df_raw[time_col] + start_system_time
    df_raw["Pressure"] = df_raw[pressure_col]
    df_merged = df_raw[["Time", "Pressure"]].copy().sort_values("Time")

    # if merged.csv exists → merge
    output_blob = bucket_dest.blob(OUTPUT_PATH)
    if output_blob.exists():
        old_text = output_blob.download_as_text()
        df_old = pd.read_csv(StringIO(old_text))
        df_merged = pd.concat([df_old, df_merged], ignore_index=True)
        df_merged.drop_duplicates(subset=["Time"], inplace=True)
        df_merged.sort_values("Time", inplace=True)
        print(f"🧩 Merged with existing data → {len(df_merged)} rows")

    # update ZIP
    blob.upload_from_string(zip_bytes, content_type="application/zip")
    print(f"✅ Uploaded raw ZIP to gs://{BUCKET_NAME_SOURCE}/{new_name}")

    # Upload merged.csv
    csv_bytes = df_merged.to_csv(index=False, float_format="%.6f").encode("utf-8")
    output_blob.upload_from_string(csv_bytes, content_type="text/csv")
    print(f"✅ Updated merged.csv ({len(df_merged)} rows)")

    return file.filename