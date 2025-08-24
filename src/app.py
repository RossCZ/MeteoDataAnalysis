from datetime import datetime
from pathlib import Path

import pandas as pd

import db_helper as db
import other_experiments as exp
from processor import DataProcessor, Plotter, DaysCounters
from settings import GeneralSettings, ResampleSettings, ChannelSettings, Resamples, Labels, INDEX_COL, MeteoChannels, CHANNELS_AIR


def process_csv_data(
        settings: GeneralSettings,
        resamples: list[ResampleSettings],
        channels: list[ChannelSettings] | None = None,
        multi_channels: list[list[ChannelSettings]] | None = None,
):
    Path(settings.out_root).mkdir(parents=True, exist_ok=True)  # create output directory

    """Main method for processing measured data"""
    # load file once
    print("Loading file...")
    df_main = DataProcessor.load_file(settings.input_file, INDEX_COL)
    print(f"\tLoaded {len(df_main)} rows")

    # split graphs into years (year None = all years together)
    years = DataProcessor.get_data_years(df_main) if settings.split_years else [None]

    print("Processing data...")

    # cumulative plots
    cum_channels = {}
    if settings.split_years and settings.cumulative_plots:
        print("Cumulative plots")
        cum_channels["Mean"] = channels.copy()
        cum_channels["Min"] = channels.copy()
        cum_channels["Max"] = channels.copy()
        cum_channels["MinMax"] = channels.copy()  # coldest day
        cum_channels["MaxMin"] = channels.copy()  # warmest night
        if not settings.is_air:
            count_group = [
                ChannelSettings(channel_name="Sunny days", yaxis_label="Sunny days"),
                ChannelSettings(channel_name="Tropic days", yaxis_label="Tropic days (Tmax > 30°C)"),
                ChannelSettings(channel_name="Tropic nights", yaxis_label="Tropic nights (Tmin > 20°C)"),
                ChannelSettings(channel_name="Freezing days", yaxis_label="Freezing days (Tmin < 0°C)"),
                ChannelSettings(channel_name="Ice days", yaxis_label="Ice days (Tmax < 0°C)"),
                ChannelSettings(channel_name="Constant days", yaxis_label="Constant days (Tspan < 2°C)"),
                ChannelSettings(channel_name="Measurements", yaxis_label="Number of measurements")
            ]
            cum_channels["Counts"] = count_group
        cumulative_plots(settings, cum_channels, years, df_main)

    # statistics files
    print("Statistics")
    statistics_folder = Path(settings.out_root, "Statistics")
    statistics_folder.mkdir(parents=True, exist_ok=True)
    DataProcessor.create_statistics_file(df_main, channels, "All", statistics_folder, settings)
    for year in years:
        DataProcessor.create_statistics_file(DataProcessor.slice_year(year, df_main), channels, year, statistics_folder, settings, f"statistics_{year}.txt")
    # idea: overview md table for all/selected units (columns) and years (rows)

    # --- year plots ~old, may be removed in the future ---
    if settings.yearly_plots:
        for year in years:
            if year is not None and year < settings.start_year:
                continue

            print(f"Year {year}")
            year_root = settings.out_root
            if year is not None:
                year_root = Path(settings.out_root, str(year))
                year_root.mkdir(parents=True, exist_ok=True)

            df_year = DataProcessor.slice_year(year, df_main)
            DataProcessor.create_statistics_file(df_year, channels, year, year_root, settings)

            # create and plot data for each channel and each resample range separately
            for r_settings in resamples:
                print(f"\t{r_settings.label}")
                out_folder = Path(year_root, r_settings.label)
                out_folder.mkdir(parents=True, exist_ok=True)

                # counts plot
                print("\t\tCounts")
                df_counts = DataProcessor.create_counts_dataframe(df_year, r_settings.pd_resample, settings)
                Plotter.plot_data_single(df_counts, "Counts", "Measurements [-]", "green", r_settings.xticks_format, Path(out_folder, "Counts"))

                # sunny days count plot
                # if r_settings.pd_resample == Resamples.M.value.pd_resample:
                #     print("\t\tSunny")
                #     df_sunny = DataProcessor.create_sunny_counts_dataframe(df_year, r_settings.pd_resample, settings)
                #     Plotter.plot_data_single(df_sunny, "Sunny days", "Sunny days [-]", "darkorange", r_settings.xticks_format, Path(out_folder, "Sunny"))

                # one-channel plot
                if channels:
                    dfs_final = {}
                    for ch_settings in channels:
                        print(f"\t\t{ch_settings.channel_name}")
                        out_file = Path(out_folder, ch_settings.channel_name)
                        df = DataProcessor.create_interval_dataframe(df_year, r_settings.pd_resample, ch_settings.channel_key, settings)
                        dfs_final[ch_settings.channel_key] = df
                        Plotter.plot_data_single(df, ch_settings.channel_name, ch_settings.yaxis_label, ch_settings.color, r_settings.xticks_format, out_file)
                        # break

                    # save daily dataframes to the DB
                    if settings.db_out_file and r_settings.pd_resample == "D":
                        df_db = db.prepare_df_for_db(dfs_final)
                        db.write_to_db(df_db, year, settings.db_out_file)

                # multi-channel plot
                if multi_channels:
                    for i, ch_settings_list in enumerate(multi_channels, start=1):
                        name = f"Multiple {i}"
                        print(f"\t\t{name}")
                        out_file = Path(out_folder, name)
                        df = DataProcessor.create_multi_dataframe(df_year, r_settings.pd_resample, ch_settings_list, settings)
                        Plotter.plot_data_multi(df, name, ch_settings_list, r_settings.xticks_format, out_file)

    print("Finished")


def cumulative_plots(settings, channel_groups, years, df_main):
    cumulative_resample = Resamples.M.value.pd_resample

    for channel_group, channels in channel_groups.items():
        print(f"\t{channel_group}")
        ser_cum_tout_prep = None
        for ch_settings in channels:
            print(f"\t\t{ch_settings.channel_name}")

            # process group and channel
            ser_cum = None  # this ensures, that all groups and channel names are matched
            if channel_group == "Counts":
                if ser_cum_tout_prep is None:
                    ser_cum_tout_prep = DataProcessor.preprocess_day_counts_series(df_main, cumulative_resample, MeteoChannels.Tout.value.channel_key)  # speed optimization

                if ch_settings.channel_name == "Sunny days":
                    ser_cum = DataProcessor.create_day_counts_series(df_main, cumulative_resample, MeteoChannels.Tbalc.value.channel_key, DaysCounters.sunny_days)
                elif ch_settings.channel_name == "Tropic days":
                    ser_cum = ser_cum_tout_prep.apply(DaysCounters.tropic_days)
                elif ch_settings.channel_name == "Tropic nights":
                    ser_cum = ser_cum_tout_prep.apply(DaysCounters.tropic_nights)
                elif ch_settings.channel_name == "Freezing days":
                    ser_cum = ser_cum_tout_prep.apply(DaysCounters.freezing_days)
                elif ch_settings.channel_name == "Ice days":
                    ser_cum = ser_cum_tout_prep.apply(DaysCounters.ice_days)
                elif ch_settings.channel_name == "Constant days":
                    ser_cum = ser_cum_tout_prep.apply(DaysCounters.constant_days)
                elif ch_settings.channel_name == "Measurements":
                    ser_cum = DataProcessor.create_counts_dataframe(df_main, cumulative_resample, settings)["Counts"]
            else:
                ser_cum = DataProcessor.create_interval_dataframe(df_main, cumulative_resample, ch_settings.channel_key, settings, aggregation=channel_group)[channel_group]

            # split to years
            c_data = {}
            for year in years:
                ser_year = DataProcessor.slice_year(year, ser_cum)
                ser_year.index = [dt.month for dt in ser_year.index]
                ser_year.index.name = "Month"
                c_data[year] = ser_year

            # prepare out path
            # out_folder = Path(settings.out_root, "Cumulative", channel_group)
            out_folder = Path(settings.out_root, channel_group)  # no yearly processing for now, save to root output folder
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = Path(out_folder, ch_settings.channel_name)

            # plot data
            df_c = pd.DataFrame(dict([(year, ser) for year, ser in c_data.items()]))
            Plotter.plot_data_multi_cumulative(df_c, ch_settings.channel_name, ch_settings.yaxis_label, out_path)

            # special output: md file for out temperature
            if channel_group == "Mean" and ch_settings.channel_name == "Temperature out":
                with open(Path(out_folder, "Temperature.md"), "w", encoding="utf-8") as file:
                    df_m = pd.DataFrame(dict(mean=df_c.mean(axis=1), std=df_c.std(axis=1)))
                    df_m["Temperature [°C]"] = df_m.apply(lambda m: f"{m['mean']:.1f} ± {m['std']:.1f}", axis=1)
                    df_m["Temperature [°C]"].to_markdown(file)
                Plotter.plot_data_multi_cumulative_plotly(df_c, "Mean monthly temperature", ch_settings.yaxis_label, out_path)


def process_meteo():
    # yearly: daily, weekly, monthly aggregation + cumulative
    data_folder = Path("0_Data")
    out_folder = Path("0_OutputMeteo")
    """
    
split_years: bool
    input_file: Path
    out_root: Path
    start_year: int
    db_out_file: Path
    is_air: bool = False
    cumulative_plots
    """
    settings = GeneralSettings(
        split_years=True,
        input_file=Path(data_folder, "feeds_meteo.csv"),
        out_root=out_folder,
        start_year=2019,
        db_out_file=Path(out_folder, "yearly_data.db"),
        is_air=False,
        cumulative_plots=True,
        yearly_plots=False,
    )
    channels_meteo = [e.value for e in MeteoChannels]
    process_csv_data(
        settings,
        [Resamples.D.value, Resamples.W.value, Resamples.M.value],
        channels_meteo
    )

    # all: weekly, monthly, yearly aggregation
    if settings.yearly_plots:
        settings.split_years = False
        process_csv_data(
            settings,
            [Resamples.W.value, Resamples.M.value, Resamples.Y.value],
            channels_meteo
        )


def process_air():
    data_folder = Path("0_Data")
    out_folder = Path("0_OutputAir")
    settings = GeneralSettings(False, Path(data_folder, "feeds_air.csv"), out_folder, 2022, Path(out_folder, "yearly_data.db"), is_air=True)
    process_csv_data(
        settings,
        [Resamples.D.value, Resamples.W.value, Resamples.M.value, Resamples.Y.value],
        CHANNELS_AIR,
        [
            [CHANNELS_AIR[2], CHANNELS_AIR[4], CHANNELS_AIR[6]],
        ]
    )


if __name__ == "__main__":
    process_meteo()
    # process_air()

    # other experiments
    # exp.api_read()
    # exp.validate_std()
    # exp.data_exploration_1()
    # exp.show_day(Path("0_Data", "feeds_air.csv"), CHANNELS_AIR[0], datetime(year=2022, month=12, day=1).date(), is_air=True)
