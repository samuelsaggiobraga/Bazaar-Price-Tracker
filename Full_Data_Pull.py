import requests
import json
import time
import re
from datetime import datetime, timedelta


def fetch_all_data(item,
                   start=datetime(2020, 9, 10, 0, 0, 0),
                   end=datetime.now(),
                   interval_seconds=82800):

    base_url = "https://sky.coflnet.com/api/bazaar"
    interval = timedelta(seconds=interval_seconds)

    current = start
    raw_combined = []

    requests_made = 0
    max_requests = 30
    window_seconds = 10

    while current + interval <= end:
        start_str = current.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
        end_str   = (current + interval).strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")

        url = f"{base_url}/{item}/history?start={start_str}&end={end_str}"
        print("Fetching:", url)

        try:
            resp = requests.get(url)
            data = resp.json()

            if isinstance(data, list):
                raw_combined.extend(data)
            elif isinstance(data, dict):
                raw_combined.append(data)

        except Exception as e:
            print("Error:", e)

        requests_made += 1
        if requests_made >= max_requests:
            print(f"Hit {max_requests} requests → waiting {window_seconds} seconds...")
            time.sleep(window_seconds)
            requests_made = 0

        current += interval

    return raw_combined


temp = requests.get("https://sky.coflnet.com/api/items/bazaar/tags")
itemIDs = re.findall(r'"([^"]*)"', temp.text)


for itemID in itemIDs:
    all_data = fetch_all_data(itemID)


    with open(f"bazaar_history_combined_{itemID}.json", "w") as f:
        json.dump(all_data, f, indent=4)

    print("Saved", len(all_data), "entries.")
