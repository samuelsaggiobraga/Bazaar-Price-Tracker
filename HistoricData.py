import requests
import re
#https://sky.coflnet.com/api/items/bazaar/tags
def fetch_historic_data(item_id):
    url = f"https://sky.coflnet.com/api/bazaar/{item_id}/history"
    response = requests.get(url)
    html_content = response.text
    buy_data= re.findall(r'"buy":([^,]*),', html_content)
    sell_data = re.findall(r'"sell":([^,]*),', html_content)
    time_data =re.findall(r'"timestamp":"([^"]*)"', html_content)
    buyVolume_data = re.findall(r'"buyVolume":([^,]*),', html_content)
    sellVolume_data = re.findall(r'"sellVolume":([^,]*),', html_content)
    buyMovingWeek_data = re.findall(r'"buyMovingWeek":([^,]*),', html_content)
    sellMovingWeek_data = re.findall(r'"sellMovingWeek":([^},]*)},', html_content)
    maxBuy_data = re.findall(r'"maxBuy":([^,]*),', html_content)
    maxSell_data = re.findall(r'"maxSell":([^,]*),', html_content)
    minBuy_data = re.findall(r'"minBuy":([^,]*),', html_content)
    minSell_data = re.findall(r'"minSell":([^,]*),', html_content)
    years_data = []
    months_data = []
    days_data = []
    for time in time_data:
        year = time[0:4]
        month = time[5:7]
        day = time[8:10]
        years_data .append(year)
        months_data.append(month)
        days_data.append(day)

    years_int = [int(year) for year in years_data]
    months_int = [int(month) for month in months_data]
    days_int = [int(day) for day in days_data]
    buy_data_float = [float(price) for price in buy_data]
    sell_data_float = [float(price) for price in sell_data]
    buyVolume_data_float = [float(volume) for volume in buyVolume_data]
    sellVolume_data_float = [float(volume) for volume in sellVolume_data]
    buyMovingWeek_data_int = [int(moving) for moving in buyMovingWeek_data]
    sellMovingWeek_data_int = [int(moving) for moving in sellMovingWeek_data]
    maxBuy_data_float = [float(maxb) for maxb in maxBuy_data]
    maxSell_data_float = [float(maxs) for maxs in maxSell_data]
    minBuy_data_float = [float(minb) for minb in minBuy_data]
    minSell_data_float = [float(mins) for mins in minSell_data]




