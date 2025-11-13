from Bazaar import get_bazaar_buy_data, get_bazaar_sell_data
import time
import json

buy_prices = []
sell_prices = []
item_id = "BOOSTER_COOKIE"


def calculate_z_score_outliers(prices):
    mean_price = sum(prices) / len(prices)
    variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
    std_dev = variance ** 0.5
    outliers = []
    for p in prices:
        z_score = (p - mean_price) / std_dev if std_dev != 0 else 0
        if abs(z_score) > 3:
            outliers.append(p)
    return outliers

def calculate_tukey_outliers(prices):
    sorted_prices = sorted(prices)
    q1_index = len(sorted_prices) // 4
    q3_index = (len(sorted_prices) * 3) // 4
    q1 = sorted_prices[q1_index]
    q3 = sorted_prices[q3_index]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [p for p in prices if p < lower_bound or p > upper_bound]
    return outliers
def collect_prices_loop(item_id: str = item_id, interval: int = 30, max_samples: int | None = None, verbose: bool = True):
    count = 0
    try:
        while max_samples is None or count < max_samples:
            buy_price = get_bazaar_buy_data(item_id)
            sell_price = get_bazaar_sell_data(item_id)
            buy_prices.append(buy_price)
            sell_prices.append(sell_price)
            count += 1
            if verbose:
                print(f"Collected {len(buy_prices)} data points...")
            time.sleep(interval)
            if len(buy_prices) % 100 == 0:
                buy_outliers = calculate_tukey_outliers(buy_prices)
                sell_outliers = calculate_tukey_outliers(sell_prices)
                z_score_buy_outliers = calculate_z_score_outliers(buy_prices)
                z_score_sell_outliers = calculate_z_score_outliers(sell_prices)
                if verbose:
                    print(f"Tukey Buy Outliers: {buy_outliers}")
                    print(f"Tukey Sell Outliers: {sell_outliers}")
                    print(f"Z-Score Buy Outliers: {z_score_buy_outliers}")
                    print(f"Z-Score Sell Outliers: {z_score_sell_outliers}")
    except KeyboardInterrupt:
        if verbose:
            print("Collection interrupted by user")


if __name__ == "__main__":
    collect_prices_loop()