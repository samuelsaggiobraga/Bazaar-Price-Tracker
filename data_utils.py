import requests
import json
import time
from datetime import datetime, timedelta


def fetch_all_data(item, start=datetime(2020, 9, 10, 0, 0, 0), end=None, interval_seconds=82800):
    """Fetch bazaar data from API.
    
    Args:
        item: The item ID to fetch
        start: Start datetime for data collection
        end: End datetime for data collection (defaults to now)
        interval_seconds: Interval between API calls
        
    Returns:
        List of data entries from the API
    """
    if end is None:
        end = datetime.now()

    base_url = "https://sky.coflnet.com/api/bazaar"
    interval = timedelta(seconds=interval_seconds)

    current = start
    raw_combined = []

    requests_made = 0
    max_requests = 30
    window_seconds = 10

    while current + interval <= end:
        start_str = current.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
        end_str = (current + interval).strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")

        url = f"{base_url}/{item}/history?start={start_str}&end={end_str}"
        print(f"Fetching: {url}")

        try:
            resp = requests.get(url)
            data = resp.json()

            if isinstance(data, list):
                raw_combined.extend(data)
            elif isinstance(data, dict):
                raw_combined.append(data)

        except Exception as e:
            print(f"Error: {e}")

        requests_made += 1
        if requests_made >= max_requests:
            print(f"Hit {max_requests} requests → waiting {window_seconds} seconds...")
            time.sleep(window_seconds)
            requests_made = 0

        current += interval

    return raw_combined


def load_or_fetch_item_data(item_id, fetch_if_missing=True):
    """Load item data from file, fetching from API if it doesn't exist.
    
    Args:
        item_id: The item ID to load
        fetch_if_missing: Whether to fetch from API if file doesn't exist
        
    Returns:
        List of data entries, or None if file doesn't exist and fetch_if_missing is False
    """
    import os
    
    json_dir = "/Users/samuelbraga/Json Files"
    filename = os.path.join(json_dir, f"bazaar_history_combined_{item_id}.json")
    
    if not os.path.exists(filename):
        if fetch_if_missing:
            print(f"File {filename} not found! Fetching data from API...")
            print(f"This may take several minutes depending on data volume.")
            all_data = fetch_all_data(item_id)
            
            # Save the fetched data
            with open(filename, 'w') as f:
                json.dump(all_data, f, indent=4)
            
            print(f"Saved {len(all_data)} entries to {filename}")
            return all_data
        else:
            print(f"File {filename} not found!")
            return None
    
    print(f"Loading existing data from {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data
