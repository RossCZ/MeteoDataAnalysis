import os
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import db_helper as db
from dataclasses import dataclass
import other_experiments as exp


@dataclass
class GeneralSettings:
    """Class to hold general data processing settings"""
    split_years: bool
    input_file: str
    out_root: str
    start_year: int
    db_out_file: str
    is_air: bool = False


@dataclass
class ResampleSettings:
    label: str
    xticks_format: object


@dataclass
class ChannelSettings:
    channel_name: str
    yaxis_label: str
    color: str


DEFAULT_RESAMPLES = {
        "D": ResampleSettings("Daily", 7),
        "W": ResampleSettings("Weekly", "W"),
        "M": ResampleSettings("Monthly", "M"),
        "Y": ResampleSettings("Yearly", "Y")
    }

INDEX_COL = "created_at"

LABELS = {
    "T": "Temperature [°C]",
    "H": "Humidity [%]",
    "P": "Pressure [Pa]",
    "PM": "µg/m^3",
}

CHANNEL_MAP_METEO = {
    "field1": ChannelSettings("Temperature out", LABELS["T"], "red"),
    "field2": ChannelSettings("Air pressure", LABELS["P"], "magenta"),
    "field3": ChannelSettings("Temperature balcony", LABELS["T"], "red"),
    "field4": ChannelSettings("Temperature living room", LABELS["T"], "red"),
    "field5": ChannelSettings("Temperature bedroom", LABELS["T"], "red"),
    "field6": ChannelSettings("Humidity living room", LABELS["H"], "blue"),
    "field7": ChannelSettings("Humidity bedroom", LABELS["H"], "blue"),
}

CHANNEL_MAP_AIR = {
    "field1": ChannelSettings("BrnL PM10", LABELS["PM"], "red"),
    "field2": ChannelSettings("BrnL PM2_5", LABELS["PM"], "red"),
    "field3": ChannelSettings("BrnA PM10", LABELS["PM"], "red"),
    "field4": ChannelSettings("BrnA PM2_5", LABELS["PM"], "red"),
    "field5": ChannelSettings("Ost PM10", LABELS["PM"], "blue"),
    "field6": ChannelSettings("Ost PM2_5", LABELS["PM"], "blue"),
    "field7": ChannelSettings("Jes PM10", LABELS["PM"], "green"),
}


def process_csv_data(channel_map: dict[str, ChannelSettings], settings: GeneralSettings, resamples: dict[str, ResampleSettings]):
    """Main method for processing measured data"""
    # create output directory
    if not os.path.exists(settings.out_root):
        os.mkdir(settings.out_root)

    # load file once
    df_main = load_file(settings.input_file)

    # split graphs into years (year None = all years together)
    years = [None]
    if settings.split_years:
        years = get_data_years(df_main)

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
        create_statistics_file(df_year, channel_map, year, year_root, settings)

        # create and plot data for each channel and each resample range separately
        for r_key, r_settings in resamples.items():
            print(f"\t{r_settings.label}")
            out_folder = os.path.join(year_root, r_settings.label)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

            # counts plot
            print("\t\tCounts")
            df_counts = create_counts_dataframe(df_year, r_key, settings)
            plot_data(df_counts, "Counts", "Measurements [-]", "green", r_settings.xticks_format, os.path.join(out_folder, "Counts"))

            # channels plot
            dfs_final = {}
            for ch_key, ch_settings in channel_map.items():
                print(f"\t\t{ch_settings.channel_name}")
                out_file = os.path.join(out_folder, ch_settings.channel_name)
                df = create_final_dataframe(df_year, r_key, ch_key, settings)
                dfs_final[ch_key] = df
                plot_data(df, ch_settings.channel_name, ch_settings.yaxis_label, ch_settings.color, r_settings.xticks_format, out_file)
                # break

            # save daily dataframes to the DB
            if settings.db_out_file and r_key == "D":
                df_db = db.prepare_df_for_db(dfs_final)
                db.write_to_db(df_db, year, settings.db_out_file)

    print("Finished")


def load_file(file_path):
    df = pd.read_csv(file_path, index_col=INDEX_COL, converters={INDEX_COL: lambda x: date_try_parse(x)})
    return df


def filter_dict(dict, keys):
    return {key: dict[key] for key in keys}


def date_try_parse(datetime_str):
    datetime_str_parts = datetime_str.split("+")
    datetime_str, utc_change_str = datetime_str_parts[0], datetime_str_parts[1]
    utc_change = datetime.strptime(utc_change_str, "%H:%M").time()
    datetime_fin = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")

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


def create_counts_dataframe(df, resample, settings):
    df = df.resample(resample)

    df_res = pd.DataFrame()
    df_res["Counts"] = df["field5"].count()  # bedroom
    if not settings.is_air:
        df_res["Counts"] += df["field4"].count()  # living room
    return df_res


def create_final_dataframe(df, resample, data_key, settings):
    df = df[[data_key]]
    ser = cleanup_df_column(df, data_key, settings.is_air)

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


def cleanup_df_column(df, data_key, is_air):
    # cleanup data
    df = df[[data_key]].dropna()
    ser = df[data_key]
    if not is_air:
        ser = remove_peaks(ser)

    if is_air and data_key == "field1":
        # only for outside temperature in meteo
        ser = remove_sunshine(ser)

    return pd.to_numeric(ser)


def remove_peaks(ser):
    """Some Meteo measurements contain wrong values due to error -> large peaks in data."""
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
    # empirically known sunshine hours in summer (3-9 months) for outside sensor
    sunshine_min = time(hour=5, minute=0, second=0)
    sunshine_max = time(hour=9, minute=0, second=0)
    sunshine_month_start = 3
    sunshine_month_end = 9

    # assess every day for max temperature within sunshine hours
    date_today = ser.index[0].date()
    date_end = ser.index[-1].date()
    while date_today <= date_end:
        ser_day = ser[str(date_today)]
        if not ser_day.empty:
            ser_day = pd.to_numeric(ser_day)
            datetime_of_max_temp = ser_day.idxmax()
            time_of_max_temp = datetime_of_max_temp.time()
            month_of_max_temp = datetime_of_max_temp.date().month

            # remove sunshine peak only in valid month and time ranges
            if (sunshine_month_start <= month_of_max_temp <= sunshine_month_end) and (sunshine_min < time_of_max_temp < sunshine_max):
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


def plot_data(df, name, ylabel, color, xticks_format, out_file=""):
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

    if type(xticks_format) == int:
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=xticks_format))  # label each N days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    elif type(xticks_format) == str:
        plt.xticks(df.index)  # use index dates as ticks
        if xticks_format == "W":
            if len(df) < 100:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            else:  # use month formatter for large number of weeks
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        elif xticks_format == "M":
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        elif xticks_format == "Y":
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

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


def create_statistics_file(df_year, channel_map, year, year_root, settings):
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
    if not settings.is_air:
        f.write(f"Number of sunny days: {count_sunny_days(df_year['field3'])}\n")  # use balcony temperature with direct sunshine
        f.write(f"Number of freezing days (Tmax < 0 °C): {count_freezing_days(df_year['field1'])}\n")
        f.write(f"Number of tropic days (Tmin > 20 °C): {count_tropic_days(df_year['field1'])}\n")
        f.write(f"Number of constant days (Tspan < 2 °C): {count_constant_days(df_year['field1'])}\n")

    for ch_key, ch_settings in channel_map.items():
        ser = cleanup_df_column(df_year, ch_key, settings.is_air)
        f.write(f"\n{ch_settings.channel_name} ({ch_settings.yaxis_label})\n")
        f.write(f"\tEntries: {ser.count()}\n")
        f.write(f"\tMean: {ser.mean():.1f} (+-{ser.std():.1f})\n")
        f.write(f"\tMin:  {ser.min()}\t({ser.idxmin()})\n")
        f.write(f"\tMax:  {ser.max()}\t({ser.idxmax()})\n")
    f.close()


def process_meteo():
    # yearly: daily, weekly, monthly aggregation
    data_folder = "0_Data"
    out_folder = "0_OutputMeteo"
    settings = GeneralSettings(True, os.path.join(data_folder, "feeds_meteo.csv"), out_folder, 2019, os.path.join(out_folder, "yearly_data.db"))
    process_csv_data(CHANNEL_MAP_METEO, settings, filter_dict(DEFAULT_RESAMPLES, ["D", "W", "M"]))

    # all: weekly, monthly, yearly aggregation
    settings.split_years = False
    process_csv_data(CHANNEL_MAP_METEO, settings, filter_dict(DEFAULT_RESAMPLES, ["W", "M", "Y"]))


def process_air():
    data_folder = "0_Data"
    out_folder = "0_OutputAir"
    settings = GeneralSettings(False, os.path.join(data_folder, "feeds_air.csv"), out_folder, 2022, os.path.join(out_folder, "yearly_data.db"), is_air=True, )
    process_csv_data(CHANNEL_MAP_AIR, settings, filter_dict(DEFAULT_RESAMPLES, ["W", "M", "Y"]))


if __name__ == "__main__":
    process_meteo()
    process_air()

    # other experiments
    # exp.api_read()
    # exp.validate_std()
    # exp.data_exploration_1()
    # exp.show_day(os.path.join("0_Data", "feeds_air.csv"), CHANNEL_MAP_AIR, "field1", datetime(year=2022, month=12, day=1).date(), is_air=True)
