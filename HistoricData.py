import requests
#https://sky.coflnet.com/api/items/bazaar/tags

url = "https://sky.coflnet.com/api/bazaar/BOOSTER_COOKIE/history"
response = requests.get(url)
html_content = response.text
times = html_content.split('},{')
buy_prices = []
sell_prices = []
time_stamps = []
for time in times:
    index_1 = time.find("\"buy\":")
    index_2 = time.find(",\"sell\":")
    buy_price = time[index_1+6:index_2]
    sell_price = time[index_2+8:time.find(",\"sellVolume\"")]
    time_stamp_index = time.find("\"timestamp\":")
    time_stamp = time[time_stamp_index+13:time.find(",\"buyMovingWeek\":")-1]
    time_stamps.append(time_stamp)
    buy_prices.append(buy_price)
    sell_prices.append(sell_price)
time_stamps[-1] = time_stamps[-1].replace("\"", "")
sell_prices.pop(-1)
buy_prices.pop(-1)
buy_prices_float = [float(price) for price in buy_prices]
sell_prices_float = [float(price) for price in sell_prices]
years = []
months = []
days = []
for time in time_stamps:
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    years.append(year)
    months.append(month)
    days.append(day)

years_int = [int(year) for year in years]
months_int = [int(month) for month in months]
days_int = [int(day) for day in days]


def create_data_for_ML():
    data = []
    for i in range(len(buy_prices_float)):
        data.append([years_int[i], months_int[i], days_int[i], buy_prices_float[i], sell_prices_float[i]])
    return data




