from datetime import datetime, timezone, timedelta
from unicodedata import name

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








