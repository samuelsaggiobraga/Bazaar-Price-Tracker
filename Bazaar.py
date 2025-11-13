from hypixel_api_lib import Bazaar
import json

bazaar = Bazaar()

def get_bazaar_buy_data(item_id):
    product = bazaar.get_product_by_id(item_id)
    lowest_buy = product.buy_summary[0].price_per_unit
    return lowest_buy

def get_bazaar_sell_data(item_id):
    product = bazaar.get_product_by_id(item_id)
    highest_sell = product.sell_summary[0].price_per_unit
    return highest_sell

