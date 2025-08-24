from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import dates
from settings import INDEX_COL, MeteoChannels
from processor import DataProcessor, DataCleaner


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
                df[INDEX_COL] = df[INDEX_COL].apply(DataProcessor.date_try_parse)
                df.set_index(INDEX_COL, inplace=True)
                print(df)

                # visualize
                ch_settings = MeteoChannels.Tout.value
                df_plot = df[ch_settings.channel_key].dropna()
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
    input_file = Path("0_Data", "feeds_meteo.csv")
    df_main = DataProcessor.load_file(input_file, INDEX_COL)

    years = DataProcessor.get_data_years(df_main)

    for year in years:
        if year != 2019:
            continue

        df_year = DataProcessor.slice_year(year, df_main)

        for e in MeteoChannels:
            channel = e.value
            if channel.channel_key != MeteoChannels.Tbalc.value.channel_key:
                continue

            df = df_year[[channel.channel_key]]
            ser = DataCleaner.cleanup_df_column(df, channel.channel_key, False)
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
    data_file = Path("0_Data", "feeds_meteo.csv")
    date_start, date_end = "2021-06-09", "2021-06-24"
    resample = "H"
    out_ch, in_ch, in_ch_b = MeteoChannels.Tout.value, MeteoChannels.TinL.value, MeteoChannels.TinB.value  # field 1, 4, 5
    is_air = False

    df = DataProcessor.load_file(data_file, INDEX_COL)
    df = df[date_start:date_end]
    ser_out = DataCleaner.cleanup_df_column(df, out_ch.channel_key, is_air).resample(resample).mean()
    ser_in = DataCleaner.cleanup_df_column(df, in_ch.channel_key, is_air).resample(resample).mean()
    ser_in_b = DataCleaner.cleanup_df_column(df, in_ch_b.channel_key, is_air).resample(resample).mean()

    plt.figure(figsize=(15, 8))
    plt.plot(ser_out, label=out_ch.channel_name, color="red")
    plt.plot(ser_in, label=in_ch.channel_name, color="blue")
    plt.plot(ser_in_b, label=in_ch_b.channel_name, color="green")
    plt.xlabel("Date")
    plt.xlim(pd.to_datetime(date_start), pd.to_datetime(date_end) + pd.Timedelta(1, "d"))
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.ylabel(out_ch.yaxis_label)
    plt.yticks(np.arange(8, 35, 1.0))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def show_day(data_file, ch_settings, day, is_air=False):
    day_str = str(day)

    df = DataProcessor.load_file(data_file, INDEX_COL)
    ser = df.loc[day_str][ch_settings.channel_key]
    ser = ser.dropna()

    plt.title(f"{ch_settings.channel_name} {day_str}")
    plt.plot(ser, color="grey", label="original")
    plt.plot(DataCleaner.cleanup_df_column(df.loc[day_str], ch_settings.channel_key, is_air), color="blue", marker=".", label="cleaned")
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    plt.xlabel("Hour")
    plt.ylabel(ch_settings.yaxis_label)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
