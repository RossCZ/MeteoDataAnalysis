import os
import pandas as pd
import numpy as np
import yaml
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import dates
from src.app import INDEX_COL, CHANNEL_MAP_METEO
from src.app import date_try_parse, load_file, get_data_years, get_year_dataframe, cleanup_df_column


def api_read():
    with open("config.yml", "r") as stream:
        try:
            # https://www.mathworks.com/help/thingspeak/readdata.html
            config = yaml.safe_load(stream)
            no_results = 300

            query_str = f'''https://api.thingspeak.com/channels/{config["channel_id"]}/feeds.json?api_key={config["read_api_key"]}&results={no_results}&timezone=Europe/Prague'''
            res = requests.get(query_str)
            if res.status_code == 200:
                json_res = res.json()  # channel, feeds
                feeds = json_res["feeds"]

                # create dataframe from json string
                df = pd.DataFrame(eval(str(feeds)), dtype=float)

                # fix timestamp and set it as index
                df[INDEX_COL] = df[INDEX_COL].apply(date_try_parse)
                df.set_index(INDEX_COL, inplace=True)
                print(df)

                # visualize
                ch_key = "field1"
                ch_settings = CHANNEL_MAP_METEO[ch_key]
                df_plot = df[ch_key].dropna()
                plt.figure(figsize=(15, 8))
                plt.plot(df_plot, color=ch_settings.color, marker="o")
                plt.title(ch_settings.channel_name)
                plt.ylabel(ch_settings.yaxis_label)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.xticks(rotation=90)
                plt.grid()
                plt.tight_layout()
                plt.show()
        except yaml.YAMLError as exc:
            print(exc)


def validate_std():
    input_file = os.path.join("0_Data", "feeds.csv")
    df_main = load_file(input_file)

    years = get_data_years(df_main)

    for year in years:
        if year != 2019:
            continue

        df_year = get_year_dataframe(year, df_main)

        for data_key, _ in CHANNEL_MAP_METEO.items():
            if data_key != "field3":
                continue

            df = df_year[[data_key]]
            ser = cleanup_df_column(df, data_key)
            resampled = ser.resample("D")
            ser_mean = resampled.mean()
            ser_count = resampled.count()

            values, weights = ser_mean.to_list(), ser_count.to_list()
            # weights = [1 for i in ser_count]
            avg = np.average(values, weights=weights)
            std = np.sqrt(np.average((values - avg) ** 2, weights=weights))

            print(ser_mean)
            print(ser_count)

            # https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy
            # Note: biased vs. unbiased estimator of std (defaults of numpy and pandas)
            # Calculated from the original values - base truth
            print(f"\tMean:   {ser.mean():.3f} (+-{ser.std():.3f}) (+-{np.std(ser):.3f})")
            # Calculated from daily values, not weighted - not correct
            print(f"\tMeanR:  {ser_mean.mean():.3f} (+-{np.std(ser_mean.to_list()):.3f}) (+-{ser_mean.std():.3f})")
            # Calculated from daily values, weighted - avg correct, std not
            print(f"\tMeanWR: {avg:.3f} (+-{std:.3f})")  # biased estimator (ddof = 0)


def data_exploration_1():
    data_file = os.path.join("0_Data", "feeds_meteo.csv")
    date_start, date_end = "2021-06-09", "2021-06-24"
    out_ch, in_ch, in_ch_b = "field1", "field4", "field5"
    resample = "H"

    df = load_file(data_file)
    df = df[date_start:date_end]
    ser_out = cleanup_df_column(df, out_ch).resample(resample).mean()
    ser_in = cleanup_df_column(df, in_ch).resample(resample).mean()
    ser_in_b = cleanup_df_column(df, in_ch_b).resample(resample).mean()

    channel_map = CHANNEL_MAP_METEO

    plt.figure(figsize=(15, 8))
    plt.plot(ser_out, label=channel_map[out_ch].channel_name, color="red")
    plt.plot(ser_in, label=channel_map[in_ch].channel_name, color="blue")
    plt.plot(ser_in_b, label=channel_map[in_ch_b].channel_name, color="green")
    plt.xlabel("Date")
    plt.xlim(pd.to_datetime(date_start), pd.to_datetime(date_end) + pd.Timedelta(1, "d"))
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.ylabel(channel_map[out_ch].yaxis_label)
    plt.yticks(np.arange(8, 35, 1.0))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def show_day(data_file, channel_map, ch_key, day, is_air=False):
    day_str = str(day)
    ch_settings = channel_map[ch_key]

    df = load_file(data_file)
    ser = df.loc[day_str][ch_key]
    ser = ser.dropna()

    plt.title(f"{ch_settings.channel_name} {day_str}")
    plt.plot(ser, color="grey", label="original")
    plt.plot(cleanup_df_column(df.loc[day_str], ch_key, is_air), color="blue", marker=".", label="cleaned")
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    plt.xlabel("Hour")
    plt.ylabel(ch_settings.yaxis_label)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
