from Utils.data_utils import load_or_fetch_item_data
import requests
import json
import statistics

url = "https://sky.coflnet.com/api/items/bazaar/tags"
item_ids = requests.get(url).json()
items = []
for item_id in item_ids:
    total = 0
    count = 0
    total_price = 0
    data = load_or_fetch_item_data(item_id)
    for entry in data:
        if not isinstance(entry, dict) or 'buyVolume' not in entry or 'buy' not in entry:
            continue
        Volume = entry['buyVolume']
        total += Volume
        count += 1
    if count == 0:
        continue
    prices = [entry['buy'] for entry in data if isinstance(entry, dict) and 'buy' in entry]
    if not prices:
        continue
    median_price = statistics.median(prices)
    Average_Volume = total/count
    volume_price_weight = Average_Volume * median_price
    if median_price > 100000 and Average_Volume > 5000:
        items.append({
            'item_id': item_id,
            'volume_price_weight': volume_price_weight
        })
items.sort(key=lambda x: x['volume_price_weight'], reverse=True)
with open('sorted_by_demand_items.json', 'w') as f:
        json.dump(items, f, indent=1)

        

    


