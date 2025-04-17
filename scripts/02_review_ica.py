# %%
# package imports
from sys import argv

import mne
import pathlib

from importlib import reload
import shared_functions as sfuns
import matplotlib as mpl

reload(sfuns)

# %%
# config

mpl.use('macosx')

# project file path
project_path = pathlib.Path(__file__).parents[1]

# raw data filepath
raw_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/raw")

# epochs filepath
epochs_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/epochs")

# ICA results path
ica_path = project_path / "results" / "ica"

# autoreject objects path
ar_path = project_path / "results" / "autoreject"

# select participant

# set manually
# p = 1

# or set via command line argument
p = int(argv[1])

# %%
# read epochs
epochs = mne.read_epochs(epochs_path / f"{p}" / f"{p}-raw-epo.fif", preload=True)

# read raw data for EOG epochs and filter as for preprocessing
raw = sfuns.read_raw(p, raw_path=raw_path, project_path=project_path)
sfuns.apply_filters(raw, .2, 30, 50)

# load ICA
ica = mne.preprocessing.read_ica(ica_path / f"{p}-ica.fif")

# %%

# plotting diagnostics

# find which ICs match the EOG pattern
eog_epochs = mne.preprocessing.create_eog_epochs(raw)
eog_evoked = eog_epochs.average()
eog_indices, eog_scores = ica.find_bads_eog(epochs)
ecg_indices, ecg_scores = ica.find_bads_ecg(epochs, ch_name="Heart")

# start with automatically detected EOG and ECG components
ica.exclude = eog_indices + ecg_indices

ica_done = False

while not ica_done:
    # plot ICA components
    ica.plot_components(inst=epochs)

    # plot overlay
    ica.plot_overlay(raw, exclude=ica.exclude)

    # barplot of ICA component "EOG match" scores
    ica.plot_scores(eog_scores)

    # barplot of ICA component "ECG match" scores
    ica.plot_scores(ecg_scores, exclude=ecg_indices, title="ECG components")

    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    ica.plot_sources(eog_evoked)

    print(f'\nCurrent picks are {ica.exclude}. You happy? (y/n)')
    if input() == 'y':
        print(f'Fine, saving ICA with picks {ica.exclude}.')
        ica_done = True

# %%
# save ICA solution with updated component picks

# save ica solution to file
ica.save(ica_path / f'{p}-reviewed-ica.fif', overwrite=True)
