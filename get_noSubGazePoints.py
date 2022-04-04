from matplotlib import pyplot as plt
import pickle
import pandas as pd
import numpy as np
from numpy.random._common import interface

from utils import gazedata_helper_dataclasses


def get_target_gaze_participants():
    return ['003_data.tsv', '007_data.tsv', '008_data.tsv', '0012_data.tsv', '0013_data.tsv', '0016_data.tsv', '0018_data.tsv', '0020_data.tsv',
            '0023_data.tsv', '0024_data.tsv', '0025_data.tsv', '0026_data.tsv', '0027_data.tsv', '0028_data.tsv', '0029_data.tsv', '0030_data.tsv',
            '0031_data.tsv','0033_data.tsv','0034_data.tsv','0037_data.tsv','0041_data.tsv','0045_data.tsv','0049_data.tsv','0050_data.tsv','0052_data.tsv']

def load_picked_file(filepath: str) -> gazedata_helper_dataclasses.GroupData:
    return pickle.load(open(filepath, 'rb'))

def subset_participants_from_group(group_dataclass: gazedata_helper_dataclasses.GroupData, participants_list: list):
    subset_data = None #prepare a copy of the data frame
    
    for participant in group_dataclass.group_data:
        if participant.participant_id in participants_list:
            participant_data = participant.gaze_data
            participant_data['participant_id'] = participant.participant_id
            if subset_data is None:
                subset_data = participant_data
            else:
                subset_data = pd.concat([subset_data, participant_data])
    return subset_data


def get_labelled_participants_by_group(group_dataclass: gazedata_helper_dataclasses.GroupData, target_list: list, target_label: str, nontarget_label: str, drop_nas = True) -> (pd.DataFrame, pd.DataFrame):
    all_data = None
    for participant in group_dataclass.group_data:
        participant_data = participant.gaze_data
        participant_data['participant_id'] = participant.participant_id
        if participant.participant_id in target_list:
            participant_data['group'] = target_label
        else:
            participant_data['group'] = nontarget_label
        if all_data is None:
            all_data = participant_data
        else:
            all_data = pd.concat([all_data,participant_data])

    if drop_nas:
        all_data = all_data[(all_data['x'].isna().__invert__()) | (all_data['y'].isna().__invert__())]
    return all_data

def plot_gaze_scatter(df1: pd.DataFrame, df2: pd.DataFrame, label1, label2, scene):
    fig = plt.figure()
    plt.scatter(df1[df1['scene'] == scene]['x'], 1080 - df1[df1['scene'] == scene]['y'], label = label1, alpha=0.1)
    plt.scatter(df2[df2['scene'] == scene]['x'], 1080 - df2[df2['scene'] == scene]['y'], label=label2, alpha=0.1)
    plt.legend()
    fig.axes[0].set_xlim((0, 1920))
    fig.axes[0].set_ylim((0, 1080))
    plt.savefig("/home/vishnu/Desktop/temp/" + label1 + "_" + label2 + "_" + str(scene) + ".png")

if __name__ == "__main__":
    german_participants = get_target_gaze_participants()
    group_data_noSubs = load_picked_file('./pickles/noSub.pkl')
    group_data_tSubs = load_picked_file("./pickles/tSub.pkl")
    group_data_iSubs = load_picked_file("./pickles/iSub.pkl")

    noSubs_lang = get_labelled_participants_by_group(group_data_noSubs, german_participants, 'German', 'nonGerman')
    noSubs_lang = noSubs_lang[((noSubs_lang['x'].isna()) | (noSubs_lang['y'].isna())).__invert__()]

    tSubs_lang = get_labelled_participants_by_group(group_data_tSubs, german_participants, 'German','nonGerman')
    iSubs_lang = get_labelled_participants_by_group(group_data_iSubs, german_participants,'German','nonGerman')

    plot_gaze_scatter(iSubs_lang[iSubs_lang['group'] == 'German'], iSubs_lang[iSubs_lang['group'] == 'nonGerman'], 'german', 'nonGerman', 50)

    plot_gaze_scatter(tSubs_lang, iSubs_lang,'tSubs', 'iSubs', 50)
    plot_gaze_scatter(noSubs_lang, iSubs_lang, 'noSubs', 'iSubs', 50)
    plot_gaze_scatter(noSubs_lang, tSubs_lang, 'noSubs', 'tSubs', 50)
    # scene_number = 50
    # fig = plt.figure()
    # plt.scatter(german_gaze[(german_gaze['group']=='german') & (german_gaze['scene'] == scene_number)]['x'],
    #             1080 - german_gaze[(german_gaze['group']=='german') & (german_gaze['scene'] == scene_number)]['y'], alpha=0.1, label = 'German')
    # plt.scatter(german_gaze[(german_gaze['group'] == 'nonGerman') & (german_gaze['scene'] == scene_number)]['x'],
    #             1080 - german_gaze[(german_gaze['group'] == 'nonGerman') & (german_gaze['scene'] == scene_number)]['y'],
    #             alpha=0.1, label = 'nonGerman')
    # fig.axes[0].set_xlim((0,1920))
    # fig.axes[0].set_ylim((0,1080))
    # plt.legend()
    # fig = plt.figure()
    # plt.scatter(x=scene_data['x'], y=scene_data['y'])
    # plt.title("Scene " + str(scene))
    # plt.show()
    # input("press any key to continue")

'''
so that makes 4 tests:
nonGermans-tSubs v/s nonGermans-noSubs (filter only for tSubs People)
nonGerman-tsubs v/s German-tsubs (filter only for all tSubs people)
german-Tsubs v/s german-noSubs (filter only for tSubsGermans)
nonGerman-tsubs v/s nonGermans-noSubs

==================================================

nonGermans-isubs v/s germans-noSubs
IS LESS THAN
nonGermans-tSubs v/s germans-NoSubs


Germans-iSubs v/s Germans-NoSubs 
IS LESS THAN 
Germans-tSubs v/s Germans-noSubs




===================
interesting scenes:
2,3,12,15,39,41,44,73,75,77,79,81,83,85,89,92,94,96
'''

interesting_scenes = [2,3,12,15,39,41,44,73,75,77,79,81,83,85,89,92,94,96]
iSubs_participants = []
for participant in group_data_iSubs.group_data:
    iSubs_participants.append(participant.participant_id)

tSub_participants = []
for participant in group_data_tSubs.group_data:
    tSub_participants.append(participant.participant_id)

for scene in interesting_scenes:
    tSubs_lang[tSubs_lang.participant_id.isin(german_participants).__invert__()]


people = tSubs_lang.participant_id.unique()
nonGerman_tSubs = [person for person in people if person not in german_participants]
german_tSubs = [person for person in people if person in german_participants]

people = iSubs_lang.participant_id.unique()
nonGerman_iSubs = [person for person in people if person not in german_participants]
german_iSubs = [person for person in people if person in german_participants]

for scene in interesting_scenes[:5]:
    plot_gaze_scatter(tSubs_lang[tSubs_lang.participant_id.isin(nonGerman_tSubs)], noSubs_lang[noSubs_lang.participant_id.isin(nonGerman_tSubs)], 'tSubs', 'noSubs', scene)


plot_gaze_scatter(tSubs_lang[tSubs_lang.participant_id.isin(nonGerman_tSubs)], noSubs_lang[noSubs_lang.participant_id.isin(nonGerman_tSubs)], 'tSubs', 'noSubs', 15)

plot_gaze_scatter(tSubs_lang[tSubs_lang.participant_id.isin(nonGerman_tSubs)], iSubs_lang[iSubs_lang.participant_id.isin(nonGerman_iSubs)], 'tSubs_ng', 'iSubs_ng', 2)
plot_gaze_scatter(tSubs_lang[tSubs_lang.participant_id.isin(nonGerman_tSubs)], iSubs_lang[iSubs_lang.participant_id.isin(nonGerman_iSubs)], 'tSubs_ng', 'iSubs_ng', 3)
plot_gaze_scatter(tSubs_lang[tSubs_lang.participant_id.isin(nonGerman_tSubs)], iSubs_lang[iSubs_lang.participant_id.isin(nonGerman_iSubs)], 'tSubs_ng', 'iSubs_ng', 12)
plot_gaze_scatter(tSubs_lang[tSubs_lang.participant_id.isin(nonGerman_tSubs)], iSubs_lang[iSubs_lang.participant_id.isin(nonGerman_iSubs)], 'tSubs_ng', 'iSubs_ng', 15)


#for scene in interesting_scenes[:5]:
#    #plot_gaze_scatter(tSubs_lang, iSubs_lang, 'tSubs_all', 'iSubs_all', scene)
    #plot_gaze_scatter(iSubs_lang, noSubs_lang, 'iSubs_all', 'noSubs_all', scene)
    plot_gaze_scatter(tSubs_lang, noSubs_lang, 'tSubs_all', 'noSubs_all', scene)
