from dataclasses import dataclass
import pandas as pd

@dataclass
class CalibrationData:
    cal_sample_time: float
    cal_accuracy: dict
    cal_precision: dict
    cal_fixation_threshold: float
    cal_speed_threshold: float
    cal_accuracy_threshold: float
    cal_display_distance: int

    cal_sample_rate: int = 60
    display_resolution: tuple = (1920, 1080)
    display_size: tuple = (33.8, 27.1)
    fixation_threshold: float = 1.5
    speed_threshold: float = 35
    acceleration_threshold: float = 9500

@dataclass
class ParticipantData:
    participant_id: str
    calibration_data: CalibrationData
    gaze_data: pd.DataFrame


@dataclass
class GroupData:
    group_name: str
    group_data: list
