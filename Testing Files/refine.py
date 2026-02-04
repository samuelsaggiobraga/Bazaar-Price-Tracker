import json
import gzip
import pickle
import os

BIG_DIR = os.path.join(os.path.dirname(__file__), 'bazaar_data')
OUTPUT_FILE = 'bazaar_full_items_ids.json'

with open('sorted_by_demand_items.json', 'r') as f:
    items = json.load(f)

valid_item_ids = []

for entry in items:
    item_id = entry['item_id']

    with gzip.open(f'{BIG_DIR}/bazaar_history_{item_id}.pkl.gz', 'rb') as f:
        item_data = pickle.load(f)

    # Determine number of entries
    if isinstance(item_data, dict):
        num_entries = len(next(iter(item_data.values())))
    elif isinstance(item_data, list):
        num_entries = len(item_data)
    else:
        raise TypeError(f"Unexpected item_data type: {type(item_data)}")

    if num_entries >= 200000:
        valid_item_ids.append(item_id)
    else:
        print(f"Insufficient data for item: {item_id}")

# Save only the item IDs to JSON
with open(OUTPUT_FILE, 'w') as f:
    json.dump(valid_item_ids, f, indent=2)

print(f"Saved {len(valid_item_ids)} valid item IDs to {OUTPUT_FILE}")
