import numpy
import pandas as pd

import config
from utils.gazedata_helper_dataclasses import *
import os
import copy

def blink_detection(x, y, time, missing=0.0, minlen=10):
    """Detects blinks, defined as a period of missing data that lasts for at least a minimal amount of samples

	Parameters
	----------
	x : array
		Gaze x positions
	y :	array
		Gaze y positions
	time : array
		Timestamps
	missing	: float
		Value to be used for missing data (default = 0.0)
	minlen : int
		Minimal amount of consecutive missing samples

	Returns
	-------
	Sblk : list of lists
		Each containing [starttime]
	Eblk : list of lists
		Each containing [starttime, endtime, duration]

	"""

    # empty list to contain data
    Sblk = []
    Eblk = []

    # check where the missing samples are
    mx = numpy.array(x == missing, dtype=int)
    my = numpy.array(y == missing, dtype=int)
    miss = numpy.array((mx + my) == 2, dtype=int)

    # check where the starts and ends are (+1 to counteract shift to left)
    diff = numpy.diff(miss)
    starts = numpy.where(diff == 1)[0] + 1
    ends = numpy.where(diff == -1)[0] + 1

    # compile blink starts and ends
    for i in range(len(starts)):
        # get starting index
        s = starts[i]
        # get ending index
        if i < len(ends):
            e = ends[i]
        elif len(ends) > 0:
            e = ends[-1]
        else:
            e = -1
        # append only if the duration in samples is equal to or greater than
        # the minimal duration
        if e - s >= minlen:
            # add starting time
            Sblk.append([time[s]])
            # add ending time
            Eblk.append([time[s], time[e], time[e] - time[s]])

    return Sblk, Eblk


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):
    """Detects fixations, defined as consecutive samples with an inter-sample distance of less than a set amount of pixels (disregarding missing data)

	Parameters
	----------
	x : array
		Gaze x positions
	y :	array
		Gaze y positions
	time : array
		Timestamps
	missing	: float
		Value to be used for missing data (default = 0.0)
	maxdist : int
		Maximal inter sample distance in pixels (default = 25)
	mindur : int
		Minimal duration of a fixation in milliseconds; detected fixation cadidates will be disregarded if they are below this duration (default = 100)

	Returns
	-------
	Sfix : list of lists
		Each containing [starttime]
	Efix : list of lists
		Each containing [starttime, endtime, duration, endx, endy]

	"""

    # empty list to contain data
    Sfix = []
    Efix = []

    # loop through all coordinates
    si = 0
    fixstart = False
    for i in range(1, len(x)):
        # calculate Euclidean distance from the current fixation coordinate
        # to the next coordinate
        dist = ((x[si] - x[i]) ** 2 + (y[si] - y[i]) ** 2) ** 0.5
        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            Sfix.append([time[i]])
        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i - 1] - Sfix[-1][0] >= mindur:
                Efix.append([Sfix[-1][0], time[i - 1], time[i - 1] - Sfix[-1][0], x[si], y[si]])
            # delete the last fixation start if it was too short
            else:
                Sfix.pop(-1)
            si = 0 + i
        elif not fixstart:
            si += 1

    return Sfix, Efix


def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):
    """Detects saccades, defined as consecutive samples with an inter-sample velocity of over a velocity threshold or an acceleration threshold

	Parameters
	----------
	x : array
		Gaze x positions
	y :	array
		Gaze y positions
	time : array
		Timestamps
	missing	: float
		Value to be used for missing data (default = 0.0)
	minlen : int
		Minimal length of saccades in milliseconds; all detected saccades with len(sac) < minlen will be ignored (default = 5)
	maxvel : int
		Velocity threshold in pixels/second (default = 40)
	maxacc : int
		Acceleration threshold in pixels / second**2 (default = 340)

	Returns
	-------
	Ssac : list of lists
		Each containing [starttime]
	Esac : list of lists
		Each containing [starttime, endtime, duration, startx, starty, endx, endy]

	"""

    # CONTAINERS
    Ssac = []
    Esac = []

    # INTER-SAMPLE MEASURES
    # the distance between samples is the square root of the sum
    # of the squared horizontal and vertical interdistances
    intdist = (numpy.diff(x) ** 2 + numpy.diff(y) ** 2) ** 0.5
    # get inter-sample times
    inttime = numpy.diff(time)
    # recalculate inter-sample times to seconds
    inttime = inttime / 1000.0

    # VELOCITY AND ACCELERATION
    # the velocity between samples is the inter-sample distance
    # divided by the inter-sample time
    vel = intdist / inttime
    # the acceleration is the sample-to-sample difference in
    # eye movement velocity
    acc = numpy.diff(vel)

    # SACCADE START AND END
    t0i = 0
    stop = False
    while not stop:
        # saccade start (t1) is when the velocity or acceleration
        # surpass threshold, saccade end (t2) is when both return
        # under threshold

        # detect saccade starts
        sacstarts = numpy.where((vel[1 + t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
        if len(sacstarts) > 0:
            # timestamp for starting position
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time) - 1:
                t1i = len(time) - 2
            t1 = time[t1i]

            # add to saccade starts
            Ssac.append([t1])

            # detect saccade endings
            sacends = numpy.where((vel[1 + t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
            if len(sacends) > 0:
                # timestamp for ending position
                t2i = sacends[0] + 1 + t1i + 2
                if t2i >= len(time):
                    t2i = len(time) - 1
                t2 = time[t2i]
                dur = t2 - t1

                # ignore saccades that did not last long enough
                if dur >= minlen:
                    # add to saccade ends
                    Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                else:
                    # remove last saccade start on too low duration
                    Ssac.pop(-1)

                # update t0i
                t0i = 0 + t2i
            else:
                stop = True
        else:
            stop = True

    return Ssac, Esac


def get_gaze_data(filename: str, gaze_metadata: CalibrationData):
    """Returns a list with dicts for every trial.

    Parameters
    ----------
    filename : str
    	Path to the file that has to be read
    start : str
    	Trial start string
    stop : str
    	Trial ending string (default = None)
    missing : float
    	Value to be used for missing data (default = 0.0)
    debug : bool
    	Indicating if DEBUG mode should be on or off; if DEBUG mode is on, information on what the script currently is doing will be printed to the console (default = False)

    Returns
    -------
    data : list
    	With a dict for every trial. Following is the dictionary
    	0. x -array of Gaze x positions,
    	1. y -array of Gaze y positions,
    	2. size -array of pupil size,
    	3. time -array of timestamps, t=0 at trialstart,
    	4. trackertime -array of timestamps, according to the tracker,
    	5. events -dict {Sfix, Ssac, Sblk, Efix, Esac, Eblk, msg}

    """
    debug = True
    if debug:
        def message(msg):
            print(msg)
    else:
        def message(msg):
            pass

    # # # # #
    # file handling

    # check if the file exists
    if os.path.isfile(filename):
        # open file
        message("opening file '%s'" % filename)
        f = open(filename, 'r')
    # raise exception if the file does not exist
    elif os.path.isfile(filename.split("/")[-1]):
        message("opening file '%s'" % filename)
        f = open(filename.split("/")[-1], 'r')
    else:
        raise Exception("Error in read_tobii: file '%s' does not exist" % filename)

    # read file contents
    message("reading file '%s'" % filename)
    raw = f.readlines()

    # close file
    message("closing file '%s'" % filename)
    f.close()

    # # # # #
    # parse lines

    # variables
    data = []
    x_l = []
    y_l = []
    x_r = []
    y_r = []
    size_l = []
    size_r = []
    trackertime = []
    events = {'Sfix': [], 'Ssac': [], 'Sblk': [], 'Efix': [], 'Esac': [], 'Eblk': [], 'msg': []}
    starttime = 0
    started = False
    trialend = False
    filestarted = False

    timei = None
    msgi = -1
    eventi = -1
    xi = {'L': None, 'R': None}
    yi = {'L': None, 'R': None}
    sizei = {'L': None, 'R': None}
    legacy = False

    # Skip lines until headers
    header_row_index = 0
    while raw[header_row_index].__contains__("TimeStamp") == False:
        header_row_index += 1

    # Found header row. Set up indices.
    header_line = raw[header_row_index].replace('\n', '').replace('\r', '').split('\t')
    timei = header_line.index("TimeStamp")
    eventi = header_line.index("Event")
    xi = {'L': None, 'R': None}
    yi = {'L': None, 'R': None}
    sizei = {'L': None, 'R': None}
    xi['L'] = header_line.index("GazePointXLeft")
    xi['R'] = header_line.index("GazePointXRight")
    yi['L'] = header_line.index("GazePointYLeft")
    yi['R'] = header_line.index("GazePointYRight")
    sizei['L'] = header_line.index("PupilSizeLeft")
    sizei['R'] = header_line.index("PupilSizeRight")

    # Now skip forward the lines until msg Start is found
    data_row_index = header_row_index
    while raw[data_row_index].__contains__("START") == False:
        data_row_index += 1

    # loop through all lines
    missing = -1
    for i in range(data_row_index + 1, len(raw), 1):
        line = raw[i].replace('\n', '').replace('\r', '').split('\t')

        if line[1][:3] == 'END':
            break

        vi = None
        for ind, var in enumerate([xi, yi, sizei]):
            vi = var
            val_l = -1
            val_r = -1
            if float(line[vi['R']]) == -1:
                line[vi['R']] = str(missing)
            if float(line[vi['L']]) == -1:
                line[vi['L']] = str(missing)
            val_r = float(line[vi['R']])
            val_l = float(line[vi['L']])

            if ind == 0:
                x_l.append(val_l)
                x_r.append(val_r)
            elif ind == 1:
                y_l.append(val_l)
                y_r.append(val_r)
            elif ind == 2:
                size_l.append(val_l)
                size_r.append(val_r)

        trackertime.append(float(line[timei]))

        # read entire file, now create supp. info.
    trial = {}
    trial['x_l'] = numpy.array(x_l)
    trial['y_l'] = numpy.array(y_l)
    trial['x_r'] = numpy.array(x_r)
    trial['y_r'] = numpy.array(y_r)
    trial['size_l'] = numpy.array(size_l)
    trial['size_r'] = numpy.array(size_r)
    trial['trackertime'] = numpy.array([float(x) for x in trackertime])
    trial['events'] = copy.deepcopy(events)
    # TODO here is the place where i need to plug in the params from the calibration.
    trial['events']['Sblk'], trial['events']['Eblk'] = blink_detection(trial['x_l'], trial['y_l'],
                                                                       trial['trackertime'],
                                                                       missing=missing)
    trial['events']['Sfix'], trial['events']['Efix'] = fixation_detection(trial['x_l'], trial['y_l'],
                                                                          trial['trackertime'], mindur=100,
                                                                          maxdist=gaze_metadata.cal_fixation_threshold,
                                                                          missing=missing)
    trial['events']['Ssac'], trial['events']['Esac'] = saccade_detection(trial['x_l'], trial['y_l'],
                                                                         trial['trackertime'],
                                                                         missing=missing)
    # add trial to data
    data.append(trial)

    df = convert_data_to_df(data)
    return df, trial['events']


def scene_index(x, scenes_timestamps):
    # if x < scenes_timestamps[0]:
    #     return 1
    # if x >= scenes_timestamps[-1]:
    #     return len(scenes_timestamps) + 1
    for i in range(len(scenes_timestamps)):
        if x <= scenes_timestamps[i]:
            return i+1
    return len(scenes_timestamps)+1


def get_scenes(df):
    scenes_timestamps = pd.to_timedelta(open(config.get_scenes_filepath(),'r').readline().split(","))
    df['Timestamp'] = pd.to_timedelta(df['Timestamp'] * 1000000)
    scenes = df['Timestamp'].apply(lambda x: scene_index(x,scenes_timestamps))
    return scenes


def convert_data_to_df(data):
    headers = ["Timestamp", "GazeLeftx", "GazeRightx", "GazeLefty", "GazeRighty", "PupilLeft", "PupilRight",
               "FixationSeq", "SaccadeSeq", "Blink", "GazeAOI"]
    df = pd.DataFrame(columns=headers)
    i = 0
    for d in data:
        temp_dict = dict.fromkeys(headers)

        temp_dict['Timestamp'] = d['trackertime']

        '''Maybe this can be replaced with the filename or something, then we have a safe way to track the person whose data we are checking... if filepath, then we can have person,group        '''
        # if stim_list == None:
        #     temp_dict['StimulusName'] = ['stimulus_' + str(i)] * len(temp_dict['Timestamp'])
        # else:
        #     temp_dict['StimulusName'] = [stim_list[i]] * len(temp_dict['Timestamp'])

        temp_dict['EventSource'] = ['ET'] * len(temp_dict['Timestamp'])
        temp_dict['GazeLeftx'] = d['x_l']
        temp_dict['GazeRightx'] = d['x_r']
        temp_dict['GazeLefty'] = d['y_l']
        temp_dict['GazeRighty'] = d['y_r']
        temp_dict['PupilLeft'] = d['size_l']
        temp_dict['PupilRight'] = d['size_r']
        temp_dict['FixationSeq'] = numpy.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['SaccadeSeq'] = numpy.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['Blink'] = numpy.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['GazeAOI'] = numpy.ones(len(temp_dict['Timestamp'])) * -1

        fix_cnt = 0
        sac_cnt = 0
        prev_end = 0
        for e in d['events']['Efix']:
            ind_start = numpy.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = numpy.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['FixationSeq'][ind_start: ind_end + 1] = fix_cnt
            fix_cnt += 1
            if prev_end != ind_start:
                temp_dict['SaccadeSeq'][prev_end + 1: ind_start + 1] = sac_cnt
                sac_cnt += 1
            prev_end = ind_end

        cnt = 0
        for e in d['events']['Eblk']:
            try:
                ind_start = numpy.where(temp_dict['Timestamp'] == e[0])[0][0]
                ind_end = numpy.where(temp_dict['Timestamp'] == e[1])[0][0]
                temp_dict['Blink'][ind_start: ind_end + 1] = cnt
                cnt += 1
            except Exception as exception:
                print("Error found in formatBridge toBase: " + str(exception))

        df = df.append(pd.DataFrame.from_dict(temp_dict, orient='index').transpose(), ignore_index=True, sort=False)
        df['x'],df['y'] = get_gaze_xy(df)
        df['scene'] = get_scenes(df)
        return df


def get_gaze_xy(df: pd.DataFrame):
    df = df.copy()
    left_x = df['GazeLeftx']
    right_x = df['GazeRightx']
    left_y = df['GazeLefty']
    right_y = df['GazeRighty']
    left_x[left_x < 0] = numpy.nan
    left_y[left_y < 0] = numpy.nan
    right_x[right_x < 0] = numpy.nan
    right_y[right_y <0] = numpy.nan

    gaze_x = pd.DataFrame([left_x,right_x]).mean()
    gaze_y = pd.DataFrame([right_x, right_y]).mean()

    return gaze_x,gaze_y
