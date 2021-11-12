import os
from datetime import datetime, timedelta, time, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import dates


def get_resamples():
    resamples = {
        "D": "Daily",
        "W": "Weekly",
        "M": "Monthly"
    }
    return resamples


def get_data_map():
    temperature_label = "Temperature [°C]"
    humidity_label = "Humidity [%]"
    pressure_label = "Pressure [Pa]"

    data_map = {
        "field1": (2, "Temperature out", temperature_label, "red"),
        "field2": (3, "Air pressure", pressure_label, "magenta"),
        "field3": (4, "Temperature balcony", temperature_label, "red"),
        "field4": (5, "Temperature living room", temperature_label, "red"),
        "field5": (6, "Temperature bedroom", temperature_label, "red"),
        "field6": (7, "Humidity living room", humidity_label, "blue"),
        "field7": (8, "Humidity bedroom", humidity_label, "blue"),
    }
    return data_map


def main():
    # settings
    split_years = True
    input_file = os.path.join("0_Data", "feeds.csv")
    out_root = "0_Output"
    start_year = 2021

    # create output directory
    if not os.path.exists(out_root):
        os.mkdir(out_root)

    # load file once
    df_main = load_file(input_file)

    # show_day(df_main, "field1", date(year=2020, month=6, day=14))
    # return

    # split graphs into years (year None = all years together)
    years = [None]
    if split_years:
        years = get_data_years(df_main)

    data_map = get_data_map()

    # process data
    print("Processing data...")
    for year in years:
        if year < start_year:
            continue

        print(f"Year {year}")
        year_root = out_root
        if year is not None:
            year_root = os.path.join(out_root, str(year))
            if not os.path.exists(year_root):
                os.mkdir(year_root)

        df_year = get_year_dataframe(year, df_main)
        create_statistics_file(df_year, data_map, year, year_root)

        # create and plot data for each channel and each resample range separately
        for r_key, r_value in get_resamples().items():
            print(f"\t{r_value}")
            out_folder = os.path.join(year_root, r_value)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

            for dm_key, dm_value in data_map.items():
                channel_name = dm_value[1]
                print(f"\t\t{channel_name}")
                df = create_final_dataframe(df_year, r_key, dm_key)
                out_file = os.path.join(out_folder, channel_name)
                plot_data(df, channel_name, dm_value[2], dm_value[3], out_file)
                # break
    print("Finished")


def load_file(file_path):
    index_col = "created_at"
    df = pd.read_csv(file_path, index_col=index_col, converters={index_col: lambda x: date_try_parse(x)})
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


def create_final_dataframe(df, resample, data_key):
    df = df[[data_key]]
    ser = cleanup_df_column(df, data_key)

    # resample according to settings
    df_res = pd.DataFrame()
    resampled = ser.resample(resample)
    df_res["Mean"] = resampled.mean()
    df_res["Min"] = resampled.min()
    df_res["Max"] = resampled.max()

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


def count_sunny_days(ser):
    counter = 0

    date_today = ser.index[0].date()
    date_end = ser.index[-1].date()
    while date_today <= date_end:
        ser_day = ser[str(date_today)]
        if not ser_day.empty:
            title = "Cloudy day"
            # sunny day is considered when standard deviation of daily temperature is greater than 4 °C
            if ser_day.std() > 4:
                counter += 1
                title = "Sunny day"

            # calibration plot
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


def plot_data(df, name, ylabel, color, out_file=""):
    # print settings
    dpi = 100
    width = 1920 / dpi
    height = 1080 / dpi

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.96)

    for i, column in enumerate(df.columns):
        alpha = 1.0 if i == 0 else 0.3
        marker = "" if len(df) > 1 else "o"
        plt.plot(df[column], color=color, alpha=alpha, marker=marker)

    plt.title(name)
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    margin = timedelta(days=0) if len(df) > 1 else timedelta(days=7)
    plt.xlim(df.iloc[0].name - margin, df.iloc[-1].name + margin)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=7.0))  # label each 7 days
    plt.ylabel(ylabel)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # minor ticks
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle='--', alpha=0.3)
    plt.tight_layout()
    if out_file == "":
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()


def create_statistics_file(df_year, data_map, year, year_root):
    f = open(os.path.join(year_root, "statistics.txt"), "w")
    year_name = "All" if year is None else str(year)
    start_date = df_year.iloc[0].name
    end_date = df_year.iloc[-1].name
    delta = end_date - start_date

    f.write(f"Statistics for year: {year_name}\n\n")
    f.write(f"from {start_date} to {end_date}\n")
    f.write(f"Total number of entries: {len(df_year)}\n")
    f.write(f"Number of days: {delta.days}\n")
    f.write(f"Number of sunny days: {count_sunny_days(df_year['field3'])}\n")  # use balcony temperature with direct sunshine

    for dm_key, dm_value in data_map.items():
        ser = cleanup_df_column(df_year, dm_key)
        f.write(f"\n{dm_value[1]} ({dm_value[2]})\n")
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


if __name__ == "__main__":
    main()

