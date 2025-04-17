# %%
import mne
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %%
# filepath to evokeds pickle
evokeds_fpath = '../results/evokeds/all.p'

# participants list
participants = list(range(1, 27)) + list(range(29, 38))

# channel picks for plotting
channel_picks = ["F3", "Fz", "F4", "FC1", "FC2"]

# unpickle the evokeds
with open(evokeds_fpath, 'rb') as f:
    evokeds = pickle.load(f)
# %%
# ORN for inharmonic
evokeds['orn'] = {
    'inharmonic_standard': {},
    'changing_standard': {},
    'inharmonic_deviant': {},
    'changing_deviant': {},
    }
for p in participants:
    h = evokeds['harmonic']['standard'][p]
    i = evokeds['inharmonic']['standard'][p]
    c = evokeds['inharmonic_changing']['standard'][p]
    evokeds['orn']['inharmonic_standard'][p] = mne.combine_evoked([i, h], weights=[1, -1])
    evokeds['orn']['changing_standard'][p] = mne.combine_evoked([c, h], weights=[1, -1])
    h = evokeds['harmonic']['deviant_1'][p]
    i = evokeds['inharmonic']['deviant_1'][p]
    c = evokeds['inharmonic_changing']['deviant_1'][p]
    evokeds['orn']['inharmonic_deviant'][p] = mne.combine_evoked([i, h], weights=[1, -1])
    evokeds['orn']['changing_deviant'][p] = mne.combine_evoked([c, h], weights=[1, -1])
# %%
comp = {
    'harmonic standard': list(evokeds['harmonic']['standard'].values()),
    'inharmonic standard': list(evokeds['inharmonic']['standard'].values()),
    'difference (ORN)': list(evokeds['orn']['inharmonic_standard'].values())
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine='mean')
# %%
comp = {
    'harmonic standard': list(evokeds['harmonic']['standard'].values()),
    'changing standard': list(evokeds['inharmonic_changing']['standard'].values()),
    'difference (ORN)': list(evokeds['orn']['changing_standard'].values())
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine='mean')
# %%
comp = {
    'harmonic deviant': list(evokeds['harmonic']['deviant_1'].values()),
    'inharmonic deviant': list(evokeds['inharmonic']['deviant_1'].values()),
    'difference (ORN)': list(evokeds['orn']['inharmonic_deviant'].values())
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine='mean')
# %%
comp = {
    'harmonic deviant': list(evokeds['harmonic']['deviant_1'].values()),
    'changing deviant': list(evokeds['inharmonic_changing']['deviant_1'].values()),
    'difference (ORN)': list(evokeds['orn']['changing_deviant'].values())
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine='mean')
# %%
comp = {
    'standard harmonic vs inharmonic': list(evokeds['orn']['inharmonic_standard'].values()),
    'standard harmonic vs changing': list(evokeds['orn']['changing_standard'].values()),
    'deviant harmonic vs inharmonic': list(evokeds['orn']['inharmonic_deviant'].values()),
    'deviant harmonic vs changing': list(evokeds['orn']['changing_deviant'].values()),
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine='mean')
# direct comparison of ORN waves

# %%
