import requests
import json
import re
from data_utils import fetch_all_data


temp = requests.get("https://sky.coflnet.com/api/items/bazaar/tags")
itemIDs = re.findall(r'"([^"]*)"', temp.text)

print(len(itemIDs))
for itemID in itemIDs:
    all_data = fetch_all_data(itemID)


    with open(f"bazaar_history_combined_{itemID}.json", "w") as f:
        json.dump(all_data, f, indent=4)

    print("Saved", len(all_data), "entries.")
