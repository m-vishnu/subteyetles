folder_path = r'./data/noSub_data'
pygaze_outfile_ext = r'.tsv'

eye_tracker_type = r'tobii'
eye_tracker_serial_number = r'ABCDEFGHIJKL0000'

group_data_paths = {
    'noSub': './data/noSub',
    'iSub': './data/iSub',
    'tSub': './data/tSub'
}

group_data_scenes_info = {
    'noSub': './data/noSub/scene_list.txt',
    'iSub': './data/iSub/scene_list.txt',
    'tSub': './data/iSub/scene_list.txt' 
}

def get_scenes_filepath():
    return r'./data/scene_list.txt'
