from data_utils import load_or_fetch_item_data
import requests
import json

url = "https://sky.coflnet.com/api/items/bazaar/tags"
item_ids = requests.get(url).json()
items = []
for item_id in item_ids:
    total = 0
    count = 0
    data = load_or_fetch_item_data(item_id)
    for entry in data:
        if not isinstance(entry, dict) or 'buyVolume' not in entry:
            continue
        Volume = entry['buyVolume']
        total += Volume
        count += 1
    if count == 0:
        continue
    Average_Volume = total/count
    items.append({
        'item_id': item_id,
        'Average Volume': Average_Volume
    })
items.sort(key=lambda x: x['Average Volume'], reverse=True)
with open('sorted_by_demand_items.json', 'w') as f:
        json.dump(items, f, indent=1)

        

    


