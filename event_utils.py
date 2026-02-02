from datetime import datetime, timezone, timedelta
from unicodedata import name
import os

epoch = datetime(2019, 6, 11, 17, 55, tzinfo=timezone.utc)
time_to_new_year= timedelta(hours = 124)
time_to_season_of_jerry= timedelta(hours = 113, minutes = 40)
time_to_jerry_festival = timedelta(hours = 121, minutes =40)
time_to_spooky_festival = timedelta(hours = 89, minutes = 20)

def find_skyblock_year(timestamp):
    global epoch
    global time_to_new_year

    delta = timestamp - epoch
    years_passed = delta // time_to_new_year
    current_year = 1 + years_passed
    return current_year

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

epoch = pd.Timestamp("2019-06-11 17:55", tz="UTC")

time_to_new_year = pd.Timedelta(hours=124)
time_to_season_of_jerry = pd.Timedelta(hours=113, minutes=40)
time_to_jerry_festival = pd.Timedelta(hours=121, minutes=40)
time_to_spooky_festival = pd.Timedelta(hours=89, minutes=20)


def add_skyblock_time_features(df, ts_col="timestamp"):
    dt = pd.to_datetime(df[ts_col], utc=True)

    delta = dt - epoch
    years_passed = delta // time_to_new_year
    current_year_start = epoch + years_passed * time_to_new_year

    season_of_jerry_start = current_year_start + time_to_season_of_jerry
    season_of_jerry_end = current_year_start + time_to_new_year
    jerry_festival_start = current_year_start + time_to_jerry_festival
    jerry_festival_end = jerry_festival_start + pd.Timedelta(hours=1)
    spooky_festival_start = current_year_start + time_to_spooky_festival
    spooky_festival_end = spooky_festival_start + pd.Timedelta(hours=1)


    df["time_to_season_of_jerry_start"] = (
        dt - season_of_jerry_start
    ).dt.total_seconds() / 60.0

    df["time_to_season_of_jerry_end"] = (
        dt - season_of_jerry_end
    ).dt.total_seconds() / 60.0

    df["time_to_jerry_festival_start"] = (
        dt - jerry_festival_start
    ).dt.total_seconds()

    df["time_to_jerry_festival_end"] = (
        dt - jerry_festival_end
    ).dt.total_seconds()

    df["time_to_spooky_festival_start"] = (
        dt - spooky_festival_start
    ).dt.total_seconds()

    df["time_to_spooky_festival_end"] = (
        dt - spooky_festival_end
    ).dt.total_seconds()

    df["is_during_season_of_jerry"] = (
        (dt >= season_of_jerry_start) & (dt < season_of_jerry_end)
    ).astype(int)

    df["is_during_jerry_festival"] = (
        (dt >= jerry_festival_start) & (dt < jerry_festival_end)
    ).astype(int)

    df["is_during_spooky_festival"] = (
        (dt >= spooky_festival_start) & (dt < spooky_festival_end)
    ).astype(int)


    minutes = dt.dt.minute

    minutes_to_prev_dark = (minutes - 55) % 60
    minutes_to_next_dark = (55 - minutes) % 60

    df["minutes_to_prev_dark_auction"] = minutes_to_prev_dark
    df["minutes_to_next_dark_auction"] = minutes_to_next_dark
    df["time_delta_from_dark_auction"] = np.minimum(
        minutes_to_prev_dark, minutes_to_next_dark
    )



    minutes_to_prev_jc = (minutes - 15) % 60
    minutes_to_next_jc = (15 - minutes) % 60

    df["minutes_to_prev_jacob_contest"] = minutes_to_prev_jc
    df["minutes_to_next_jacob_contest"] = minutes_to_next_jc
    df["time_delta_from_jacob_contest"] = np.minimum(
        np.abs(minutes_to_prev_jc), np.abs(minutes_to_next_jc)
    )

    return df








#Non Vectorized
def get_important_time_information(timestamp):
    global epoch
    global time_to_new_year
    global time_to_jerry_festival

    current_year =find_skyblock_year(timestamp)
    current_year_start = epoch + (current_year - 1) * time_to_new_year
    season_of_jerry_start = current_year_start + time_to_season_of_jerry
    season_of_jerry_end = current_year_start + time_to_new_year
    jerry_festival_start = current_year_start + time_to_jerry_festival
    jerry_festival_end = jerry_festival_start + timedelta(hours =1)
    spooky_festival_start = current_year_start + time_to_spooky_festival
    spooky_festival_end = spooky_festival_start + timedelta(hours =1)
    
    current_minute = timestamp.minute
    current_hour = timestamp.hour



    prev_hour = current_hour if current_minute >= 55 else current_hour - 1
    next_hour = current_hour if current_minute < 55 else current_hour + 1
  
    minutes_to_prev_dark_auction = (current_hour - prev_hour) * 60 + (current_minute - 55)
    minutes_to_next_dark_auction = (next_hour - current_hour) * 60 + (55 - current_minute)
    time_delta_from_dark_auction = min(abs(minutes_to_prev_dark_auction), abs(minutes_to_next_dark_auction))

    prev_hour = current_hour if current_minute >= 15 else current_hour - 1
    next_hour = current_hour if current_minute < 15 else current_hour + 1

    minutes_to_prev_jacob_contest = (current_hour - prev_hour) * 60 + (current_minute - 15)
    minutes_to_next_jacob_contest = (next_hour - current_hour) * 60 + (15 - current_minute)
    time_delta_from_jacob_contest = min(abs(minutes_to_prev_jacob_contest), abs(minutes_to_next_jacob_contest))


    return season_of_jerry_start, season_of_jerry_end, jerry_festival_start, jerry_festival_end, spooky_festival_start, spooky_festival_end, minutes_to_prev_jacob_contest, minutes_to_next_jacob_contest, time_delta_from_jacob_contest, minutes_to_prev_dark_auction, minutes_to_next_dark_auction, time_delta_from_dark_auction


# Testing Logic
if name == "__main__":
    now = datetime.now(timezone.utc)
    print(now)
    current_year = find_skyblock_year(now)
    season_of_jerry_start, season_of_jerry_end, jerry_festival_start, jerry_festival_end, spooky_festival_start, spooky_festival_end, minutes_to_prev_jacob_contest, minutes_to_next_jacob_contest, time_delta_from_jacob_contest, minutes_to_prev_dark_auction, minutes_to_next_dark_auction, time_delta_from_dark_auction = get_important_time_information(now)
    print(f"Current SkyBlock Year: {current_year}")
    print(f"Season of Jerry Start: {season_of_jerry_start}")
    print(f"Season of Jerry End: {season_of_jerry_end}")
    print(f"Jerry Festival Start: {jerry_festival_start}")
    print(f"Jerry Festival End: {jerry_festival_end}")
    print(f"Spooky Festival Start: {spooky_festival_start}")
    print(f"Spooky Festival End: {spooky_festival_end}")
    print(f"Minutes to Previous Jacob Contest: {minutes_to_prev_jacob_contest}")
    print(f"Minutes to Next Jacob Contest: {minutes_to_next_jacob_contest}")
    print(f"Time Delta from Jacob Contest: {time_delta_from_jacob_contest}")
    print(f"Minutes to Previous Dark Auction: {minutes_to_prev_dark_auction}")
    print(f"Minutes to Next Dark Auction: {minutes_to_next_dark_auction}")
    print(f"Time Delta from Dark Auction: {time_delta_from_dark_auction}")
