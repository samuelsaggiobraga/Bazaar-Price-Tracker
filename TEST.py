import requests
import time
import datetime
import json

# === SETTINGS ===
ITEM = "BOOSTER_COOKIE"
INTERVAL_SECONDS = 3600  # one hour (set to 60*5 for 5 minutes, etc.)
START_DATE = datetime.datetime(2020, 9, 9, tzinfo=datetime.timezone.utc)
OUTPUT_FILE = "booster_cookie_history.json"

# === MAIN ===
base_url = f"https://sky.coflnet.com/api/bazaar/{ITEM}/history/hour"
end_time = datetime.datetime.now(datetime.timezone.utc)
all_data = []

while end_time > START_DATE:
    start_time = end_time - datetime.timedelta(seconds=INTERVAL_SECONDS)

    params = {
        "from": int(start_time.timestamp() * 1000),  
        "to": int(end_time.timestamp() * 1000),
    }

    try:
        r = requests.get(base_url, params=params)
        r.raise_for_status()
        data = r.json()

        if data:
            all_data.extend(data)

        print(f"Fetched {len(data)} records from {start_time} to {end_time}")
    except Exception as e:
        print(f"Error fetching {start_time}: {e}")
        time.sleep(1)

    end_time = start_time
    time.sleep(0.2)  # be nice to API

# Save results
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Saved {len(all_data)} total data points to {OUTPUT_FILE}")
