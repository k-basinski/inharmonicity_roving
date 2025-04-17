# %%
# package imports

import mne
import pathlib
from sys import argv
from os import system
from autoreject import AutoReject

from importlib import reload
import shared_functions as sfuns

reload(sfuns)
# %%

# config

# project file path
project_path = pathlib.Path("/Users/kbas/cloud/sci/harmonicity_roving")

# raw data filepath
raw_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/raw")

# epochs filepath
epochs_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/epochs")

# ICA results path
ica_path = project_path / "results" / "ica"

# autoreject objects path
ar_path = project_path / "results" / "autoreject"

# how many cores to use? -1 for all available
n_jobs = -1

# select participant

# set manually
# p = 1
# or set via command line argument
p = int(argv[1])

# read epochs
epochs = mne.read_epochs(epochs_path / f"{p}" / f"{p}-raw-epo.fif", preload=True)

# load ICA solution
ica = mne.preprocessing.read_ica(ica_path / f"{p}-reviewed-ica.fif")

# apply ICA
epochs_after_ica = ica.apply(epochs)

# %%
# apply autoreject again, this time to the actual data and save
print("\n\n Running autoreject 2nd time...\n")

ar2 = AutoReject(n_interpolate=[1, 2, 3, 4], n_jobs=n_jobs)
epochs_ar_2, reject_log_2 = ar2.fit_transform(epochs_after_ica, return_log=True)

# store autoreject reject log 1 in a file
reject_log_2.save(ar_path / f"{p}-rl2.npz", overwrite=True)



# re-reference to mastoids
epochs_ar_2.set_eeg_reference(["M1", "M2"])

# baseline correct the epochs
epochs_ar_2.apply_baseline((-0.1, 0))

# save to filepath
epochs_ar_2.save(epochs_path / f"{p}" / f"{p}-epo.fif", overwrite=True)

print(f"Participant {p} saved successfully.")
