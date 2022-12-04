from enum import Enum
from dataclasses import dataclass


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
    pd_resample: str
    label: str
    xticks_format: object


@dataclass
class ChannelSettings:
    channel_key: str
    channel_name: str
    yaxis_label: str
    color: str


class Labels(Enum):
    T = "Temperature [°C]"
    H = "Humidity [%]"
    P = "Pressure [Pa]"
    PM = "µg/m^3"


class Resamples(Enum):
    D = ResampleSettings("D", "Daily", 7)  # minimal resample with current code
    W = ResampleSettings("W", "Weekly", "W")
    M = ResampleSettings("M", "Monthly", "M")
    Y = ResampleSettings("Y", "Yearly", "Y")


INDEX_COL = "created_at"

CHANNELS_METEO = [
    ChannelSettings("field1", "Temperature out", Labels.T.value, "red"),
    ChannelSettings("field2", "Air pressure", Labels.P.value, "magenta"),
    ChannelSettings("field3", "Temperature balcony", Labels.T.value, "red"),
    ChannelSettings("field4", "Temperature living room", Labels.T.value, "red"),
    ChannelSettings("field5", "Temperature bedroom", Labels.T.value, "red"),
    ChannelSettings("field6", "Humidity living room", Labels.H.value, "blue"),
    ChannelSettings("field7", "Humidity bedroom", Labels.H.value, "blue"),
]

CHANNELS_AIR = [
    ChannelSettings("field1", "BrnL PM10", Labels.PM.value, "red"),
    ChannelSettings("field2", "BrnL PM2_5", Labels.PM.value, "red"),
    ChannelSettings("field3", "BrnA PM10", Labels.PM.value, "red"),
    ChannelSettings("field4", "BrnA PM2_5", Labels.PM.value, "red"),
    ChannelSettings("field5", "Ost PM10", Labels.PM.value, "blue"),
    ChannelSettings("field6", "Ost PM2_5", Labels.PM.value, "blue"),
    ChannelSettings("field7", "Jes PM10", Labels.PM.value, "green"),
]
