# %%
# package imports

import numpy as np
import pandas as pd
import mne
import pathlib
from sys import argv
from autoreject import AutoReject

# %%

# DEFINE FUNCTIONS

def extract_digits(x: int):
    """This function takes an int and returns a list of its digits."""
    # convert to strig and slice
    digits = list(str(x))
    return [int(digit) for digit in digits]


def int_to_frequency(n: int):
    """Converts a event trigger number to stimulus frequency."""
    res = (50 * n)
    return str(res)

# %%

# config

# project file path
project_path = pathlib.Path(__file__).parents[1]

# raw data filepath
raw_path = pathlib.Path('/Users/kbas/sci_data/harmonicity_data/raw')

# save epochs to filepath
epochs_path = pathlib.Path('/Users/kbas/sci_data/harmonicity_data/epochs')

# ICA results path
ica_path = project_path / 'results' / 'ica'

# autoreject objects path
ar_path = project_path / 'results' / 'autoreject'

# select participant

# set manually
# p = 8

# or set via command line argument
p = int(argv[1])

# should I run autoreject?
run_autoreject = True

# should I save epochs?
save_epochs = True

# set aliases for electrode positions
# for participants 1 and 2 (mistake in channel naming)
if p == 2:
    montage_aliases = {
        'O2': 'Oz',
        'O3': 'O2',
        'MastL': 'M1',
        'MastR': 'M2'
    }
elif p == 1:
    montage_aliases = {
        'O2': 'Oz',
        'O3': 'O2',
    }
else:
    montage_aliases = {
        # 'O2': 'Oz',
        # 'O3': 'O2',
        'MastL': 'M1',
        'MastR': 'M2'
    }


# load fpaths
# participant info table
participant_info = pd.read_csv(project_path / 'metadata' / 'participant_info.csv', index_col='pid')
filepaths = participant_info['eeg_path_local']

# load data
eeg_path = str(raw_path) + participant_info["eeg_path_local"][p]
raw = mne.io.read_raw_brainvision(
    raw_path /eeg_path,
    eog=['CanEye', 'LowEye'],
    misc=['Heart'],
    preload=True
    )

# downsample for participant 1
if p == 1:
    raw = raw.resample(1000)

if p < 3:
    mne.rename_channels(raw.info, montage_aliases)
# set montage
raw.set_montage('standard_1020')

# notch filter line noise
raw.notch_filter(np.arange(50, 251, 50))

# band-pass
raw.filter(l_freq=.2, h_freq=30)


# %%
# Prepare triggers and epoch the signal

# drop first five triggers of each block
ann = raw.annotations
t_diffs = []
for i in range(len(ann)):
    t = ann[i]['onset']
    if i == 0:
        tdiff = 100
    else:
        tdiff = t - ann[i-1]['onset']
    t_diffs.append([i, tdiff])

# get block starts
diffs = np.array(t_diffs)
block_starts = np.nonzero(diffs[:, 1] > 1)[0]

drop_ids = []
for b in block_starts:
    if b == 0:
        drop_ids.append(0)
    else:
        for i in range(b, b+5):
            drop_ids.append(i)

raw.annotations.delete(drop_ids)


# extract trigger data
events_from_annot, event_dict = mne.events_from_annotations(raw)

# code the 2nd and 3rd deviant
# as well as pitch differences
ce = events_from_annot[:, 2]

# extract pitch-only array
pitch_events = ce % 10

# calculate 1st-order absolute difference
pitch_diff = np.abs(np.diff(pitch_events))
# assume 0 for first stimulus
pitch_diffs = np.zeros(ce.shape, dtype=np.int8)
pitch_diffs[1:] = pitch_diff


# look for 1st order deviants
first_devs = (ce > 100)

# make second order deviants by rolling, erase the first element after roll
second_devs = np.roll(first_devs, 1)
second_pitch_diffs = np.roll(pitch_diffs, 1)
# make sure 0/False is at 0 index
second_devs[0] = False
second_pitch_diffs[0] = 0

# make third order deviants by rolling the second-order
# deviants, erase the first element after roll
third_devs = np.roll(second_devs, 1)
third_pitch_diffs = np.roll(second_pitch_diffs, 1)
third_devs[0] = False
third_pitch_diffs[0] = 0

# add up deviant codes and plug back into event array
custom_events = ce + (second_devs * 200) + (third_devs * 300)
custom_events = (custom_events-pitch_events) + pitch_diffs + second_pitch_diffs + third_pitch_diffs
events_from_annot[:, 2] = custom_events


# custom event_id
event_values = set(custom_events)
custom_event_keys = []
for event in event_values:
    event_l = extract_digits(event)
    
    # if len = 3 then it's a deviant
    if len(event_l) < 3:
        event_tag = 'standard'
    elif event_l[-3] == 1:
        event_tag = 'deviant_1'
    elif event_l[-3] == 2:
        event_tag = 'deviant_2'
    elif event_l[-3] == 3:
        event_tag = 'deviant_3'
    
    # digit -2 determines condition
    if event_l[-2] == 1:
        event_tag += '/harmonic/'
    elif event_l[-2] == 2:
        event_tag += '/inharmonic/'
    elif event_l[-2] == 3:
        event_tag += '/inharmonic_changing/'

    # digit -1 determines frequency
    freq = int_to_frequency(event_l[-1])
    event_tag += freq
    
    custom_event_keys.append(event_tag)

custom_event_dict = dict(map(lambda i,j : (i,j) , custom_event_keys, event_values))

# make epochs
epochs = mne.Epochs(
    raw=raw, 
    events=events_from_annot,
    event_id=custom_event_dict,
    tmin=-0.1, tmax=0.45,
    baseline=None, # don't baseline correct (yet)
    preload=True
    )

# %%
# Run autoreject (first before the ICA)

ar = AutoReject(n_interpolate=[1, 2, 3, 4], n_jobs=-1)
ar.fit(epochs)
ar.save(ar_path / f'{p}-ar.hdf5', overwrite=True)
# %%
epochs_ar, reject_log_1 = ar.transform(epochs, return_log=True)

# store autoreject reject log 1 in a file
reject_log_1.save(ar_path / f"{p}-rl1.npz", overwrite=True)

# %%
# ICA

# compute ica
ica = mne.preprocessing.ICA(
    n_components=None, 
    max_iter='auto', 
    random_state=97
    )

# fit ICA to ar-ed data
ica.fit(epochs_ar)

# save ica solution to file
ica.save(ica_path / f'{p}-ica.fif', overwrite=True)

# save "raw" epochs (without any preprocessing)
epochs.save(epochs_path / f"{p}" / f"{p}-raw-epo.fif", overwrite=True)
# %%
