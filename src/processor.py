from pathlib import Path
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import plotly.express as px
from plotly.express.colors import sample_colorscale
import plotly.graph_objects as go
from settings import MeteoChannels

# plt.style.use("ggplot")


class DataProcessor:
    @staticmethod
    def load_file(file_path, index_col):
        df = pd.read_csv(file_path, index_col=index_col, converters={index_col: lambda x: DataProcessor.date_try_parse(x)})
        return df

    @staticmethod
    def date_try_parse(datetime_str):
        datetime_str_parts = datetime_str.split("+")
        datetime_str, utc_change_str = datetime_str_parts[0], datetime_str_parts[1]
        utc_change = datetime.strptime(utc_change_str, "%H:%M").time()
        datetime_fin = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")

        # not consider daylight saving time -> all times in UTC +01:00 (winter time)
        if utc_change.hour > 1:
            datetime_fin -= timedelta(hours=1)
        return datetime_fin

    @staticmethod
    def get_data_years(df):
        year_min = df.iloc[0].name.year
        year_max = df.iloc[-1].name.year
        return [year for year in range(year_min, year_max + 1)]

    @staticmethod
    def slice_year(year, df, convert=False):
        if year is not None:
            date_from = f"{year}-1-1"
            date_to = f"{year}-12-31"
            if convert:
                date_from = pd.to_datetime(date_from)
                date_to = pd.to_datetime(date_to)
            return df[date_from:date_to]
        else:
            return df

    @staticmethod
    def create_counts_dataframe(df, resample, settings):
        df = df.resample(resample)

        df_res = pd.DataFrame()
        df_res["Counts"] = df[MeteoChannels.TinB.value.channel_key].count()  # meteo: bedroom
        if not settings.is_air:  # only one station for air
            df_res["Counts"] += df[MeteoChannels.TinL.value.channel_key].count()  # meteo: living room
        return df_res

    @staticmethod
    def create_day_counts_series(df, resample, data_key, counter) -> pd.Series:
        df = df[[data_key]]
        ser = DataCleaner.cleanup_df_column(df, data_key, is_air=False)
        return ser.resample(resample).apply(counter)

    @staticmethod
    def preprocess_day_counts_series(df, resample, data_key):
        # returns preprocessed series for given channel (same as create_day_counts_series before applying counter)
        df = df[[data_key]]
        ser = DataCleaner.cleanup_df_column(df, data_key, is_air=False)
        return ser.resample(resample)

    @staticmethod
    def create_interval_dataframe(df, resample, data_key, settings, aggregation="All"):
        df = df[[data_key]]
        ser = DataCleaner.cleanup_df_column(df, data_key, settings.is_air)

        # resample according to settings
        df_res = pd.DataFrame()
        resampled = ser.resample(resample)

        all = (aggregation == "All")
        if all or aggregation == "Mean":
            df_res["Mean"] = resampled.mean()
        elif all or aggregation == "Min":
            df_res["Min"] = resampled.min()
        elif all or aggregation == "Max":
            df_res["Max"] = resampled.max()
        elif all or aggregation == "Count":
            df_res["Count"] = resampled.count()
        elif all or aggregation == "MinMax":
            df_res["MinMax"] = resampled.apply(DaysCounters.daily_min_max)
        elif all or aggregation == "MaxMin":
            df_res["MaxMin"] = resampled.apply(DaysCounters.daily_max_min)

        return df_res

    @staticmethod
    def create_multi_dataframe(df, resample, ch_settings_list, settings):
        df_res = pd.DataFrame()

        for ch_settings in ch_settings_list:
            ser = DataCleaner.cleanup_df_column(df[[ch_settings.channel_key]], ch_settings.channel_key, settings.is_air)
            df_res[ch_settings.channel_name] = ser.resample(resample).mean()  # use means only
            # df_res[f"{ch_settings.channel_name} Min"] = ser.resample(resample).min()
            # df_res[f"{ch_settings.channel_name} Mean"] = ser.resample(resample).mean()
            # df_res[f"{ch_settings.channel_name} Max"] = ser.resample(resample).max()

        # minimal resampling is day - use only date part of datetime
        df_res.index = [inx.date() for inx in df_res.index.tolist()]

        return df_res

    @staticmethod
    def create_statistics_file(df_year, channels, year, year_root, settings, file_name="statistics.txt"):
        f = open(Path(year_root, file_name), "w")
        year_name = "All" if year is None else str(year)
        start_date = df_year.iloc[0].name
        end_date = df_year.iloc[-1].name
        delta = end_date - start_date

        f.write(f"Statistics for year: {year_name}\n\n")
        f.write(f"from {start_date} to {end_date}\n")
        timespan = (end_date - start_date)
        if settings.is_air:
            values_per_hour = 1
        else:
            values_per_hour = 2 * 60 / 10  # 2 stations, measurements per 10 minutes
        ratio = 100 * len(df_year) / ((timespan.days * 24 + timespan.seconds / 3600) * values_per_hour)
        f.write(f"Total number of entries: {len(df_year)} ({ratio:.1f} %)\n")
        f.write(f"Number of days: {delta.days}\n")

        # meteo-specific parameters
        if not settings.is_air:
            ser_balc = DataCleaner.cleanup_df_column(df_year, MeteoChannels.Tbalc.value.channel_key, settings.is_air)
            ser_tout = DataCleaner.cleanup_df_column(df_year, MeteoChannels.Tout.value.channel_key, settings.is_air)
            f.write(f"Number of sunny days: {DaysCounters.sunny_days(ser_balc)}\n")  # use balcony temperature with direct sunshine
            f.write(f"Number of tropic days (Tmax > 30°C): {DaysCounters.tropic_days(ser_tout)}\n")
            f.write(f"Number of tropic nights (Tmin > 20°C): {DaysCounters.tropic_nights(ser_tout)}\n")
            f.write(f"Number of freezing days (Tmin < 0°C): {DaysCounters.freezing_days(ser_tout)}\n")
            f.write(f"Number of ice days (Tmax < 0°C): {DaysCounters.ice_days(ser_tout)}\n")
            f.write(f"Number of constant days (Tspan < 2°C): {DaysCounters.constant_days(ser_tout)}\n")

        for ch_settings in channels:
            ser = DataCleaner.cleanup_df_column(df_year, ch_settings.channel_key, settings.is_air)
            f.write(f"\n{ch_settings.channel_name} ({ch_settings.yaxis_label})\n")
            f.write(f"\tEntries: {ser.count()}\n")
            f.write(f"\tMean: {ser.mean():.1f} (+-{ser.std():.1f})\n")
            f.write(f"\tMin:  {ser.min()}\t({ser.idxmin()})\n")
            f.write(f"\tMax:  {ser.max()}\t({ser.idxmax()})\n")
        f.close()


class DaysCounters:
    @staticmethod
    def __count_days_in_series(ser, filter_fn):
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
                # plt.savefig(Path("0_Sunny", f"{date_today}.png"))
                # plt.close()

            date_today += timedelta(days=1)

        return counter

    @staticmethod
    def sunny_days(ser):
        # sunny day is considered when standard deviation of daily temperature is greater than 4 °C
        return DaysCounters.__count_days_in_series(ser, lambda ser_day: ser_day.std() > 4)

    @staticmethod
    def tropic_days(ser):
        return DaysCounters.__count_days_in_series(ser, lambda ser_day: ser_day.max() > 30.0)

    @staticmethod
    def tropic_nights(ser):
        return DaysCounters.__count_days_in_series(ser, lambda ser_day: ser_day.min() > 20.0)

    @staticmethod
    def freezing_days(ser):
        return DaysCounters.__count_days_in_series(ser, lambda ser_day: ser_day.min() < 0.0)

    @staticmethod
    def ice_days(ser):
        return DaysCounters.__count_days_in_series(ser, lambda ser_day: ser_day.max() < 0.0)

    @staticmethod
    def constant_days(ser):
        return DaysCounters.__count_days_in_series(ser, lambda ser_day: (ser_day.max() - ser_day.min()) < 2.0)

    @staticmethod
    def daily_min_max(ser: pd.Series):
        return ser.resample("D").apply(lambda ser_day: ser_day.max()).min()

    @staticmethod
    def daily_max_min(ser: pd.Series):
        return ser.resample("D").apply(lambda ser_day: ser_day.min()).max()


class DataCleaner:
    @staticmethod
    def cleanup_df_column(df, channel_key, is_air):
        # cleanup data
        df = df[[channel_key]].dropna()
        ser = df[channel_key]

        if not is_air:
            ser = DataCleaner.remove_peaks(ser)
            if channel_key == MeteoChannels.Tout.value.channel_key:  # only for outside temperature in meteo
                ser = DataCleaner.remove_sunshine(ser)

        return pd.to_numeric(ser)

    @staticmethod
    def remove_peaks(ser):
        """Some Meteo measurements contain wrong values due to error -> large peaks in data."""
        if len(ser) < 1000:
            return ser

        # be careful not to remove valid values
        window_size = 100
        diff_limit = ser.std() * 3

        # calculate moving average of the series and fill start of the series with non nan values
        ser_ma = ser.rolling(window_size).mean()
        ser_ma.fillna(ser_ma.iloc[window_size - 1], inplace=True)

        # create compare max difference of each value of the series to moving average -> (True/False series)
        ser_diff = abs(ser_ma - ser) < diff_limit
        # replace False with nan (replacing True with 1.0 can be omitted as it implicitly converts)
        ser_diff = ser_diff.replace(False, np.nan)

        # remove peaks by combining series
        ser_res = ser_diff * ser
        ser_res.dropna(inplace=True)

        return ser_res

    @staticmethod
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
                if (sunshine_month_start <= month_of_max_temp <= sunshine_month_end) and (
                        sunshine_min < time_of_max_temp < sunshine_max):
                    # print(f"\t\tSunshine detected at {date_today} {time_of_max_temp}, temperature {ser_day.max()} °C")

                    timestamp_start = str(datetime.combine(date_today, sunshine_min))
                    timestamp_end = str(datetime.combine(date_today, sunshine_max))

                    # linear interpolation of values within the sunshine time
                    ser_sunshine = ser[timestamp_start:timestamp_end]
                    ser[timestamp_start:timestamp_end] = np.linspace(ser_sunshine.iloc[0], ser_sunshine.iloc[-1],
                                                                     len(ser_sunshine))

                    # plt.plot(ser_day, color="grey")
                    # plt.plot(ser[str(date_today)], color="blue")
                    # plt.show()
            date_today += timedelta(days=1)

        return ser


class Plotter:
    @staticmethod
    def plot_data_single(df, name, ylabel, color, xticks_format, out_file=""):
        Plotter.__initialize_plot()
        for i, column in enumerate(df.columns):
            if column != "Count":
                alpha = 1.0 if column in ["Mean", "Counts", "Sunny"] else 0.3
                marker = "" if len(df) > 1 else "o"
                plt.plot(df[column], color=color, alpha=alpha, marker=marker)
        Plotter.__plot_data_base(df, name, ylabel, xticks_format, out_file)

    @staticmethod
    def plot_data_multi(df, name, ch_settings_list, xticks_format, out_file=""):
        Plotter.__initialize_plot()
        for i, column in enumerate(df.columns):
            ch_settings = [chs for chs in ch_settings_list if chs.channel_name in column][0]  # find settings by column name
            alpha = 0.3 if "Min" in column or "Max" in column else 1.0
            marker = "" if len(df) > 1 else "o"
            plt.plot(df[column], color=ch_settings.color, alpha=alpha, marker=marker, label=column)
        plt.legend()
        Plotter.__plot_data_base(df, name, ch_settings_list[0].yaxis_label, xticks_format, out_file)

    @staticmethod
    def plot_data_multi_cumulative(df, name, ylabel, out_file=""):
        Plotter.__initialize_plot()
        cmap = matplotlib.cm.get_cmap("gist_earth_r")
        for i, column in enumerate(df.columns):
            marker = "."
            ratio = ((i + 1) / len(df.columns))
            plt.plot(df[column], color=cmap(ratio), alpha=ratio, marker=marker, label=column)
        plt.plot(df.mean(axis=1), color="firebrick", linestyle="dotted", linewidth=1, label="mean")
        if "days" in name or "nights" in name:
            plt.ylim(0, 32)
        plt.legend()
        Plotter.__plot_data_base(df, name, ylabel, "cumulative", out_file, xlabel="Month")

    @staticmethod
    def plot_data_multi_cumulative_plotly(df, name, ylabel, out_file=""):
        def format_rgba(rgb, a):
            return rgb.replace("rgb", "rgba").replace(")", f", {a})")

        samplepoints = np.linspace(0.2, 1, len(df.columns))
        cmap = sample_colorscale(px.colors.sequential.tempo, samplepoints)
        cmap = [format_rgba(c, samplepoints[i]) for i, c in enumerate(cmap)]  # add transparency
        fig = px.line(df, title=name, markers=True, color_discrete_sequence=cmap).update_layout(
            xaxis_title="Month",
            yaxis_title=ylabel,
            xaxis=dict(
                tickmode="linear",
                dtick=1,
            )
        )
        fig.add_trace(go.Scatter(x=df.index, y=df.mean(axis=1), name="mean", mode="lines",
                                 line=dict(color="firebrick", width=1, dash="dot")))
        if out_file:
            fig.write_html(f"{out_file}.html")
        else:
            fig.show()

    @staticmethod
    def __initialize_plot():
        # plot settings
        dpi = 100
        width = 1920 / dpi
        height = 1080 / dpi

        fig = plt.figure(figsize=(width, height), dpi=dpi)
        plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.96)
        return fig

    @staticmethod
    def __plot_data_base(df, name, ylabel, xticks_format, out_file="", xlabel="Date"):
        plt.title(name)
        plt.xlabel(xlabel)
        margin = timedelta(days=0) if len(df) > 1 else timedelta(days=7)

        if xticks_format == "cumulative":
            plt.xticks(df.index)
        else:
            plt.xticks(rotation=90)
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
