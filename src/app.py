from datetime import datetime
import os

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
    """Main method for processing measured data"""
    # create output directory
    if not os.path.exists(settings.out_root):
        os.mkdir(settings.out_root)

    # load file once
    df_main = DataProcessor.load_file(settings.input_file, INDEX_COL)

    # split graphs into years (year None = all years together)
    years = DataProcessor.get_data_years(df_main) if settings.split_years else [None]

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

        df_year = DataProcessor.get_year_dataframe(year, df_main)
        DataProcessor.create_statistics_file(df_year, channels, year, year_root, settings)

        # create and plot data for each channel and each resample range separately
        for r_settings in resamples:
            print(f"\t{r_settings.label}")
            out_folder = os.path.join(year_root, r_settings.label)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

            # counts plot
            print("\t\tCounts")
            df_counts = DataProcessor.create_counts_dataframe(df_year, r_settings.pd_resample, settings)
            Plotter.plot_data_single(df_counts, "Counts", "Measurements [-]", "green", r_settings.xticks_format, os.path.join(out_folder, "Counts"))

            # one-channel plot
            if channels:
                dfs_final = {}
                for ch_settings in channels:
                    print(f"\t\t{ch_settings.channel_name}")
                    out_file = os.path.join(out_folder, ch_settings.channel_name)
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
                    out_file = os.path.join(out_folder, name)
                    df = DataProcessor.create_multi_dataframe(df_year, r_settings.pd_resample, ch_settings_list, settings)
                    Plotter.plot_data_multi(df, name, ch_settings_list, r_settings.xticks_format, out_file)

    print("Finished")


def process_meteo():
    # yearly: daily, weekly, monthly aggregation
    data_folder = "0_Data"
    out_folder = "0_OutputMeteo"
    settings = GeneralSettings(True, os.path.join(data_folder, "feeds_meteo.csv"), out_folder, 2019, os.path.join(out_folder, "yearly_data.db"))
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
    data_folder = "0_Data"
    out_folder = "0_OutputAir"
    settings = GeneralSettings(False, os.path.join(data_folder, "feeds_air.csv"), out_folder, 2022, os.path.join(out_folder, "yearly_data.db"), is_air=True, )
    process_csv_data(
        settings,
        [Resamples.D.value, Resamples.W.value, Resamples.M.value, Resamples.Y.value],
        CHANNELS_AIR,
        [
            [CHANNELS_AIR[2], CHANNELS_AIR[4], CHANNELS_AIR[6]],
        ]
    )


if __name__ == "__main__":
    # process_meteo()
    process_air()

    # other experiments
    # exp.api_read()
    # exp.validate_std()
    # exp.data_exploration_1()
    # exp.show_day(os.path.join("0_Data", "feeds_air.csv"), CHANNELS_AIR[0], datetime(year=2022, month=12, day=1).date(), is_air=True)
