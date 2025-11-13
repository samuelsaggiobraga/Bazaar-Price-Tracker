from Bazaar import get_bazaar_buy_data, get_bazaar_sell_data
import time
import json

buy_prices = []
sell_prices = []
item_id = "BOOSTER_COOKIE" 

try:
    while True:
        buy_price = get_bazaar_buy_data(item_id)
        sell_price = get_bazaar_sell_data(item_id)
        buy_prices.append(buy_price)
        sell_prices.append(sell_price)
        print(f"Collected {len(buy_prices)} data points...")
        time.sleep(30)
except:
    pass

def get_prices_json():
    data = {
        "buy_prices": buy_prices,
        "sell_prices": sell_prices
    }
    with open(f"{item_id}_prices.json", "w") as f:
        json.dump(data, f, indent=4)
    return data

get_prices_json()