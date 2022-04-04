import os
from utils.gazedata_helper_dataclasses import GroupData
import pandas as pd
import config
from dataclasses import dataclass
from utils import gaze_data_loader



def load_group_data(group_name: str):
    files = [x for x in os.listdir(config.group_data_paths[group_name]) if x[-4:] == config.pygaze_outfile_ext]
    gaze_data_all = []

    for file in files:
        filepath = config.group_data_paths[group_name] + "/" + file
        current_participant_data = gaze_data_loader.load_gaze_data(filepath)
        gaze_data_all.append(current_participant_data)


    return GroupData(group_name=group_name, group_data=gaze_data_all)

