import pandas as pd
from utils.gazedata_helper_dataclasses import *
from utils.gaze_analyser import blink_detection, saccade_detection, fixation_detection, convert_data_to_df, get_gaze_data
import numpy as np
import copy
import os




def load_gaze_data(filepath: str):
    calibration_data = get_pygaze_metadata(filepath)
    gaze_data_df, event_data = get_gaze_data(filepath, calibration_data)
    gaze_data = ParticipantData(participant_id=filepath.split("/")[-1], calibration_data=calibration_data, gaze_data=gaze_data_df)
    return gaze_data


def process_event_data():
    pass


def convert_missing_gaze_data_to_nans():
    pass


def get_pygaze_metadata(filepath, device="tobii_x2_60"):
    data = open(filepath, 'r')
    line = data.readline()
    if line.replace("\n", "") == "pygaze initiation report start":
        '''
        info for metadata. Next five lines are expected to be resolution, display size, fixation threshold, speed threshold and acceleration threshold
        '''
        disp_resolution = get_resolution(data.readline())
        disp_size = get_disp_size(data.readline())
        fix_thresh = get_string_param(data.readline())
        speed_thresh = get_string_param(data.readline())
        acc_thresh = get_string_param(data.readline())

        if data.readline() == "pygaze initiation report end":
            print("Init. report read successfully")

        if data.readline().strip() == "pygaze calibration report start":
            samplerate = get_sample_rate(data.readline())

            sampletime = get_sample_time(data.readline())
            accuracy = get_accuracy(data.readline())
            precision = get_precision(data.readline())
            distance = get_string_param(data.readline(), has_units=True, datatype="int")
            fixation_thresh = get_string_param(data.readline(), has_units=True, datatype="float")
            speed_thresh2 = get_string_param(data.readline(), has_units=True, datatype="float")
            accuracy_thresh = get_string_param(data.readline(), has_units=True, datatype="float")
            return CalibrationData(display_resolution=disp_resolution, display_size=disp_size,
                                   fixation_threshold=fix_thresh, speed_threshold=speed_thresh,
                                   acceleration_threshold=acc_thresh,
                                   cal_sample_rate=samplerate, cal_sample_time=sampletime, cal_accuracy=accuracy,
                                   cal_precision=precision, cal_display_distance=distance,
                                   cal_fixation_threshold=fixation_thresh,
                                   cal_speed_threshold=speed_thresh2,
                                   cal_accuracy_threshold=accuracy_thresh)
            # return Gaze_Metadata(cal_sample_time=sampletime, cal_accuracy=accuracy_thresh, cal_precision=precision,
            #                      sample_rate=samplerate, display_resolution=disp_resolution, display_size=disp_size,
            #                      fixation_threshold=fixation_thresh, speed_threshold=speed_thresh,
            #                      acceleration_threshold=acc_thresh)
        else:
            print("Something was wrong with the file. It doesnt follow the expected logic")


def get_resolution(line: str):
    res = line.split(":")[1].strip().split("x")
    return (res[0], res[1])


def get_disp_size(line: str):
    disp_size = line.split(":")[1].strip().split("x")
    return (disp_size[0], disp_size[1])


def get_string_param(line: str, has_units=False, datatype: str = None):
    if not has_units:
        return line.split(":")[1].strip()
    else:
        if datatype is not None:
            return eval(datatype + "(" + line.split(":")[1].split(" ")[1] + ")")
        else:
            print("Has units but no datatype. Assuming float")
            return float(line.split(":")[1].split(" ")[1])


def get_sample_rate(line: str):
    return int(line.split(":")[1].replace("Hz", "").strip())


def get_sample_time(line: str):
    return float(line.split(":")[1].replace("ms", "").strip())


def get_accuracy(line: str):
    return eval(
        "{" + line.split(":")[1].replace("L", "'L").replace("R", "'R").replace("X", "X'").replace("Y", "Y'").replace(
            "=", ":") + "}")


def get_precision(line: str):
    return eval("{" + line.split(":")[1].replace("X", "'X'").replace("Y", "'Y'").replace("=", ":") + "}")
