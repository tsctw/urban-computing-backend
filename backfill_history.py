import requests
import csv
from datetime import datetime, timezone, timedelta
import time
import os

API_KEY = ""     # <--- 放你的 API key
LAT = 53.3575
LON = -6.2783
STEP = 2 * 3600              # 2 hours

# START_TIMESTAMP = 1759284000
START_TIMESTAMP = 1759284000
END_TIMESTAMP   = 1763676000

OUTPUT_CSV = "all_data.csv"

# -------------------------------------------------
# CSV HEADER
# -------------------------------------------------
header = [
    "timestamp",
    "temperature",
    "feels_like",
    "pressure",
    "humidity",
    "dew_point",
    "uvi",
    "clouds",
    "visibility",
    "wind_speed",
    "wind_deg",
    "weather_id",
    "weather_main",
    "weather_description"
]

# -------------------------------------------------
# Step 1: Load existing CSV (if exists)
# -------------------------------------------------
existing_rows = {}

if os.path.exists(OUTPUT_CSV):
    print("📄 Loading existing CSV...")
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp"])
            existing_rows[ts] = row
else:
    print("📄 No existing CSV. Will create new file.")

# -------------------------------------------------
# FETCH LOOP
# -------------------------------------------------
current = START_TIMESTAMP

while current <= END_TIMESTAMP:
    print(f"⏳ Fetching {current} -> {datetime.utcfromtimestamp(current)} UTC")

    url = (
        f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={LAT}&lon={LON}&dt={current}&appid={API_KEY}&units=metric"
    )

    r = requests.get(url)

    if r.status_code != 200:
        print(f"❌ Error {r.status_code}: {r.text}")
        time.sleep(1)
        current += STEP
        continue

    data = r.json()

    if "data" not in data or len(data["data"]) == 0:
        print("⚠️ Response missing 'data' field")
        current += STEP
        continue

    entry = data["data"][0]
    weather = entry["weather"][0] if entry.get("weather") else {}

    row = {
        "timestamp": entry.get("dt"),
        "temperature": entry.get("temp"),
        "feels_like": entry.get("feels_like"),
        "pressure": entry.get("pressure"),
        "humidity": entry.get("humidity"),
        "dew_point": entry.get("dew_point"),
        "uvi": entry.get("uvi"),
        "clouds": entry.get("clouds"),
        "visibility": entry.get("visibility"),
        "wind_speed": entry.get("wind_speed"),
        "wind_deg": entry.get("wind_deg"),
        "weather_id": weather.get("id"),
        "weather_main": weather.get("main"),
        "weather_description": weather.get("description"),
    }

    # store/overwrite by timestamp key
    existing_rows[row["timestamp"]] = row

    current += STEP
    time.sleep(0.8)

print("✔ Fetching completed. Now sorting & writing to CSV...")

# -------------------------------------------------
# Step 3: Sort by timestamp
# -------------------------------------------------
sorted_rows = sorted(existing_rows.values(), key=lambda x: int(x["timestamp"]))


# -------------------------------------------------
# Step 4: Write final sorted CSV
# -------------------------------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(sorted_rows)

print(f"🎉 Done! Saved sorted data to {OUTPUT_CSV}")