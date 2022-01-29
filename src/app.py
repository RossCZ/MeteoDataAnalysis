import os
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import yaml
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import dates
import db_helper as db
from dataclasses import dataclass


@dataclass
class GeneralSettings:
    """Class to hold general data processing settings"""
    split_years: bool
    input_file: str
    out_root: str
    start_year: int


@dataclass
class ResampleSettings:
    label: str
    xticks_distance: int


@dataclass
class ChannelSettings:
    channel_name: str
    yaxis_label: str
    color: str


def get_default_resamples():
    resamples = {
        "D": ResampleSettings("Daily", 7),
        "W": ResampleSettings("Weekly", 7),
        "M": ResampleSettings("Monthly", 30),
        "Y": ResampleSettings("Yearly", 365)
    }
    return resamples


def get_channel_map():
    temperature_label = "Temperature [°C]"
    humidity_label = "Humidity [%]"
    pressure_label = "Pressure [Pa]"

    channel_map = {
        "field1": ChannelSettings("Temperature out", temperature_label, "red"),
        "field2": ChannelSettings("Air pressure", pressure_label, "magenta"),
        "field3": ChannelSettings("Temperature balcony", temperature_label, "red"),
        "field4": ChannelSettings("Temperature living room", temperature_label, "red"),
        "field5": ChannelSettings("Temperature bedroom", temperature_label, "red"),
        "field6": ChannelSettings("Humidity living room", humidity_label, "blue"),
        "field7": ChannelSettings("Humidity bedroom", humidity_label, "blue"),
    }
    return channel_map


def get_index_col():
    return "created_at"


def process_csv_data(settings: GeneralSettings, resamples: dict[str, ResampleSettings]):
    """Main method for processing measured data"""
    # create output directory
    if not os.path.exists(settings.out_root):
        os.mkdir(settings.out_root)

    # load file once
    df_main = load_file(settings.input_file)

    # show_day(df_main, "field1", date(year=2020, month=6, day=14))
    # return

    # split graphs into years (year None = all years together)
    years = [None]
    if settings.split_years:
        years = get_data_years(df_main)

    channel_map = get_channel_map()

    # process data
    print("Processing data...")
    for year in years:
        if year is not None and year < settings.start_year:
            continue

        print(f"Year {year}")
        year_root = settings.out_root
        if year is not None:
            year_root = os.path.join(settings.out_root, str(year))
            if not os.path.exists(year_root):
                os.mkdir(year_root)

        df_year = get_year_dataframe(year, df_main)
        create_statistics_file(df_year, channel_map, year, year_root)

        # create and plot data for each channel and each resample range separately
        for r_key, r_settings in resamples.items():
            print(f"\t{r_settings.label}")
            out_folder = os.path.join(year_root, r_settings.label)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

            # measurement counts plot
            print("\t\tCounts")
            df_counts = create_counts_dataframe(df_year, r_key)
            plot_data(df_counts, "Counts", "Measurments [-]", "green", r_settings.xticks_distance, os.path.join(out_folder, "Counts"))

            # channels plot
            dfs_final = {}
            for ch_key, ch_settings in channel_map.items():
                print(f"\t\t{ch_settings.channel_name}")
                out_file = os.path.join(out_folder, ch_settings.channel_name)
                df = create_final_dataframe(df_year, r_key, ch_key)
                dfs_final[ch_key] = df
                plot_data(df, ch_settings.channel_name, ch_settings.yaxis_label, ch_settings.color, r_settings.xticks_distance, out_file)
                # break

            # save daily dataframes to the DB
            if r_key == "D":
                df_db = db.prepare_df_for_db(dfs_final)
                db.write_to_db(df_db, year)

    print("Finished")


def load_file(file_path):
    df = pd.read_csv(file_path, index_col=get_index_col(), converters={get_index_col(): lambda x: date_try_parse(x)})
    return df


def date_try_parse(datetime_str):
    datetime_str_parts = datetime_str.split("+")
    datetime_str, utc_change_str = datetime_str_parts[0], datetime_str_parts[1]
    utc_change = datetime.strptime(utc_change_str, '%H:%M').time()
    datetime_fin = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S')

    # not consider daylight saving time -> all times in UTC +01:00 (winter time)
    if utc_change.hour > 1:
        datetime_fin -= timedelta(hours=1)
    return datetime_fin


def get_data_years(df):
    year_min = df.iloc[0].name.year
    year_max = df.iloc[-1].name.year
    return [year for year in range(year_min, year_max + 1)]


def get_year_dataframe(year, df):
    if year is not None:
        return df[f"{year}-1-1":f"{year}-12-31"]
    else:
        return df


def create_counts_dataframe(df, resample):
    df = df.resample(resample)

    df_res = pd.DataFrame()
    df_res["Counts"] = df["field5"].count()  # bedroom
    df_res["Counts"] += df["field4"].count()  # living room
    return df_res


def create_final_dataframe(df, resample, data_key):
    df = df[[data_key]]
    ser = cleanup_df_column(df, data_key)

    # resample according to settings
    df_res = pd.DataFrame()
    resampled = ser.resample(resample)
    df_res["Mean"] = resampled.mean()
    df_res["Min"] = resampled.min()
    df_res["Max"] = resampled.max()
    df_res["Count"] = resampled.count()

    # minimal resampling is day - use only date part of datetime
    df_res.index = [inx.date() for inx in df_res.index.tolist()]

    return df_res


def cleanup_df_column(df, data_key):
    # cleanup data
    df = df[[data_key]].dropna()
    ser = remove_peaks(df[data_key])

    if data_key == "field1":
        # only for outside temperature
        ser = remove_sunshine(ser)

    return pd.to_numeric(ser)


def remove_peaks(ser):
    if len(ser) < 1000:
        return ser

    # be careful not to remove valid values
    window_size = 100
    diff_limit = ser.std() * 3

    # calculate moving average of the series and fill start of the series with non nan values
    ser_ma = ser.rolling(window_size).mean()
    ser_ma.fillna(ser_ma[window_size - 1], inplace=True)

    # create compare max difference of each value of the series to moving average -> (True/False series)
    ser_diff = abs(ser_ma - ser) < diff_limit
    # replace False with nan (replacing True with 1.0 can be omitted as it implicitly converts)
    ser_diff = ser_diff.replace(False, np.nan)

    # remove peaks by combining series
    ser_res = ser_diff * ser
    ser_res.dropna(inplace=True)

    return ser_res


def remove_sunshine(ser):
    # empirically known sunshine hours in summer for outside sensor
    sunshine_min = time(hour=5, minute=0, second=0)
    sunshine_max = time(hour=9, minute=0, second=0)

    # assess every day for max temperature within sunshine hours
    date_today = ser.index[0].date()
    date_end = ser.index[-1].date()
    while date_today <= date_end:
        ser_day = ser[str(date_today)]
        if not ser_day.empty:
            ser_day = pd.to_numeric(ser_day)
            time_of_max_temp = ser_day.idxmax().time()
            if sunshine_min < time_of_max_temp < sunshine_max:
                # print(f"\t\tSunshine detected at {date_today} {time_of_max_temp}, temperature {ser_day.max()} °C")

                timestamp_start = str(datetime.combine(date_today, sunshine_min))
                timestamp_end = str(datetime.combine(date_today, sunshine_max))

                # linear interpolation of values within the sunshine time
                ser_sunshine = ser[timestamp_start:timestamp_end]
                ser[timestamp_start:timestamp_end] = np.linspace(ser_sunshine[0], ser_sunshine[-1], len(ser_sunshine))

                # plt.plot(ser_day, color="grey")
                # plt.plot(ser[str(date_today)], color="blue")
                # plt.show()
        date_today += timedelta(days=1)

    return ser


def count_days_in_series(ser, filter_fn):
    counter = 0

    date_today = ser.index[0].date()
    date_end = ser.index[-1].date()
    while date_today <= date_end:
        ser_day = ser[str(date_today)]
        if not ser_day.empty:
            if filter_fn(ser_day):
                counter += 1
                # print(str(date_today))

            # Calibration plot
            # title = "Sunny day" if filter_fn(ser_day) else "Cloudy day"
            # title += f" {date_today}: {ser_day.mean():.2f} ({ser_day.std():.2f})"
            # fig, ax = plt.subplots()
            # plt.plot(ser_day.dropna())
            # plt.title(title)
            # plt.ylim(-5, 40)
            # ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
            # plt.xticks(rotation=90)
            # plt.xlim(date_today, date_today + timedelta(days=1))
            # plt.grid()
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(os.path.join("0_Sunny", f"{date_today}.png"))
            # plt.close()

        date_today += timedelta(days=1)

    return counter


def count_sunny_days(ser):
    # sunny day is considered when standard deviation of daily temperature is greater than 4 °C
    return count_days_in_series(ser, lambda ser_day: ser_day.std() > 4)


def count_freezing_days(ser):
    return count_days_in_series(ser, lambda ser_day: ser_day.max() < 0.0)


def count_tropic_days(ser):
    return count_days_in_series(ser, lambda ser_day: ser_day.min() > 20.0)


def count_constant_days(ser):
    return count_days_in_series(ser, lambda ser_day: (ser_day.max() - ser_day.min()) < 2.0)


def plot_data(df, name, ylabel, color, xticks_distance, out_file=""):
    # print settings
    dpi = 100
    width = 1920 / dpi
    height = 1080 / dpi

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.96)

    for i, column in enumerate(df.columns):
        if column != "Count":
            alpha = 1.0 if column == "Mean" or column == "Counts" else 0.3
            marker = "" if len(df) > 1 else "o"
            plt.plot(df[column], color=color, alpha=alpha, marker=marker)

    plt.title(name)
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    margin = timedelta(days=0) if len(df) > 1 else timedelta(days=7)
    plt.xlim(df.iloc[0].name - margin, df.iloc[-1].name + margin)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=xticks_distance))  # label each N days
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.ylabel(ylabel)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # minor ticks
    plt.grid(which="major", linestyle="-")
    plt.grid(which="minor", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if out_file == "":
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()


def create_statistics_file(df_year, channel_map, year, year_root):
    f = open(os.path.join(year_root, "statistics.txt"), "w")
    year_name = "All" if year is None else str(year)
    start_date = df_year.iloc[0].name
    end_date = df_year.iloc[-1].name
    delta = end_date - start_date

    f.write(f"Statistics for year: {year_name}\n\n")
    f.write(f"from {start_date} to {end_date}\n")
    timespan = (end_date - start_date)
    values_per_hour = 2 * 60/10  # 2 stations, measurements per 10 minutes
    ratio = 100 * len(df_year) / ((timespan.days * 24 + timespan.seconds / 3600) * values_per_hour)
    f.write(f"Total number of entries: {len(df_year)} ({ratio:.1f} %)\n")
    f.write(f"Number of days: {delta.days}\n")
    f.write(f"Number of sunny days: {count_sunny_days(df_year['field3'])}\n")  # use balcony temperature with direct sunshine
    f.write(f"Number of freezing days (Tmax < 0 °C): {count_freezing_days(df_year['field1'])}\n")
    f.write(f"Number of tropic days (Tmin > 20 °C): {count_tropic_days(df_year['field1'])}\n")
    f.write(f"Number of constant days (Tspan < 2 °C): {count_constant_days(df_year['field1'])}\n")

    for ch_key, ch_settings in channel_map.items():
        ser = cleanup_df_column(df_year, ch_key)
        f.write(f"\n{ch_settings.channel_name} ({ch_settings.yaxis_label})\n")
        f.write(f"\tEntries: {ser.count()}\n")
        f.write(f"\tMean: {ser.mean():.1f} (+-{ser.std():.1f})\n")
        f.write(f"\tMin:  {ser.min()}\t({ser.idxmin()})\n")
        f.write(f"\tMax:  {ser.max()}\t({ser.idxmax()})\n")
    f.close()


def show_day(df, column_name, day):
    ser = df[str(day)][column_name]
    ser = ser.dropna()
    ser_c = cleanup_df_column(df[str(day)], column_name)

    plt.title(str(day))
    plt.plot(ser, color="grey", label="original")
    plt.plot(ser_c, color="blue", marker=".", label="cleaned")
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


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
                df[get_index_col()] = df[get_index_col()].apply(date_try_parse)
                df.set_index(get_index_col(), inplace=True)
                print(df)

                # visualize
                ch_key = "field1"
                ch_settings = get_channel_map()[ch_key]
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

        for data_key, _ in get_channel_map().items():
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


def filter_dict(dict, keys):
    return {key: dict[key] for key in keys}


if __name__ == "__main__":
    # daily, weekly, monthly resample
    settings = GeneralSettings(True, os.path.join("0_Data", "feeds.csv"), "0_Output", 2019)
    process_csv_data(settings, filter_dict(get_default_resamples(), ["D", "W", "M"]))

    # yearly resample
    settings.split_years = False
    process_csv_data(settings, filter_dict(get_default_resamples(), ["Y"]))

    # other
    # api_read()
    # validate_std()

