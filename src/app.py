from datetime import datetime
from pathlib import Path

import pandas as pd

import db_helper as db
import other_experiments as exp
from processor import DataProcessor, Plotter
from settings import GeneralSettings, ResampleSettings, ChannelSettings, Resamples, INDEX_COL, CHANNELS_METEO, CHANNELS_AIR


def process_csv_data(
        settings: GeneralSettings,
        resamples: list[ResampleSettings],
        channels: list[ChannelSettings] | None = None,
        multi_channels: list[list[ChannelSettings]] | None = None,
    ):
    Path(settings.out_root).mkdir(parents=True, exist_ok=True)  # create output directory

    """Main method for processing measured data"""
    # load file once
    df_main = DataProcessor.load_file(settings.input_file, INDEX_COL)

    # split graphs into years (year None = all years together)
    years = DataProcessor.get_data_years(df_main) if settings.split_years else [None]

    print("Processing data...")

    # cumulative plots
    if settings.split_years and settings.cumulative_plots:
        print("Cumulative plots")
        cum_channels = channels.copy()
        if not settings.is_air:
            cum_channels.append(ChannelSettings(channel_name="Sunny", yaxis_label="Sunny days [-]"))
        cumulative_plots(settings, cum_channels, years, df_main)

    # year plots
    for year in years:
        if year is not None and year < settings.start_year:
            continue

        print(f"Year {year}")
        year_root = settings.out_root
        if year is not None:
            year_root = Path(settings.out_root, str(year))
            year_root.mkdir(parents=True, exist_ok=True)

        df_year = DataProcessor.get_year_dataframe(year, df_main)
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
            if r_settings.pd_resample == Resamples.M.value.pd_resample:
                print("\t\tSunny")
                df_sunny = DataProcessor.create_sunny_counts_dataframe(df_year, r_settings.pd_resample, settings)
                Plotter.plot_data_single(df_sunny, "Sunny days", "Sunny days [-]", "darkorange", r_settings.xticks_format, Path(out_folder, "Sunny"))

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


def cumulative_plots(settings, channels, years, df_main):
    out_folder = Path(settings.out_root, "Cumulative")
    out_folder.mkdir(parents=True, exist_ok=True)

    for ch_settings in channels:
        c_data = {}
        for year in years:
            df_year = DataProcessor.get_year_dataframe(year, df_main)
            if ch_settings.channel_name == "Sunny":
                df_c_year = DataProcessor.create_sunny_counts_dataframe(df_year, Resamples.M.value.pd_resample, settings)["Sunny"]
            else:
                df_c_year = DataProcessor.create_interval_dataframe(df_year, Resamples.M.value.pd_resample, ch_settings.channel_key, settings)["Mean"]
            df_c_year.index = [dt.month for dt in df_c_year.index]
            df_c_year.index.name = "Month"
            c_data[year] = df_c_year
        df_c = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in c_data.items()]))
        Plotter.plot_data_multi_cumulative(df_c, ch_settings.channel_name, ch_settings.yaxis_label, Path(out_folder, ch_settings.channel_name))

        if ch_settings.channel_name == "Temperature out":
            Plotter.plot_data_multi_cumulative_plotly(df_c, "Mean monthly temperature", ch_settings.yaxis_label, Path(out_folder, ch_settings.channel_name))


def process_meteo():
    # yearly: daily, weekly, monthly aggregation + cumulative
    data_folder = Path("0_Data")
    out_folder = Path("0_OutputMeteo")
    settings = GeneralSettings(True, Path(data_folder, "feeds_meteo.csv"), out_folder, 2019, Path(out_folder, "yearly_data.db"))
    process_csv_data(
        settings,
        [Resamples.D.value, Resamples.W.value, Resamples.M.value],
        CHANNELS_METEO
    )

    # all: weekly, monthly, yearly aggregation
    settings.split_years = False
    process_csv_data(
        settings,
        [Resamples.W.value, Resamples.M.value, Resamples.Y.value],
        CHANNELS_METEO
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
    process_air()

    # other experiments
    # exp.api_read()
    # exp.validate_std()
    # exp.data_exploration_1()
    # exp.show_day(Path("0_Data", "feeds_air.csv"), CHANNELS_AIR[0], datetime(year=2022, month=12, day=1).date(), is_air=True)
