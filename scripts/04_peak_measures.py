# %%
# extract peak measures from epochs and prepare for stats

import pathlib
import pickle
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# %%
# config

# project file path
project_path = pathlib.Path("/Users/kbas/cloud/sci/harmonicity_roving")

# plots patch
plots_path = project_path / "results" / "plots"

# epochs filepath
epochs_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/epochs")

# channel picks
channel_picks = ["Fz", "F3", "F4", "FC1", "FC2", "Cz"]

# participants list
participants = list(range(1, 27)) + list(range(29, 38))


# %%


def extract_evokeds(epochs, participants):
    epochs_d = {}
    for p in participants:
        # form path to epochs
        # fpath = epochs_path / f"{p}" / f"{p}-epo.fif"
        epochs_d[p] = epochs[f"pid == {p}"]

    evoked = {
        "harmonic": {},
        "inharmonic": {},
        "inharmonic_changing": {},
    }

    conds = ["harmonic", "inharmonic", "inharmonic_changing"]
    oddballs = ["standard", "deviant_1", "deviant_2", "deviant_3"]

    # calculate averages
    for cond in conds:
        for oddb in oddballs:
            evoked[cond][oddb] = {}
            evoked[cond][f"{oddb}_data"] = {}
            for p in participants:
                evoked[cond][oddb][p] = epochs_d[p][f"{oddb}/{cond}"].average()
                # plug raw data for cluster-based stats as well
                evoked[cond][f"{oddb}_data"][p] = evoked[cond][oddb][p].get_data()

    # calculate mismatches
    for cond in conds:
        for i in range(1, 4):
            evoked[cond][f"mismatch_{i}"] = {}
            evoked[cond][f"mismatch_{i}_data"] = {}
            for p in participants:
                d = evoked[cond][f"deviant_{i}"][p]
                s = evoked[cond]["standard"][p]
                evoked[cond][f"mismatch_{i}"][p] = mne.combine_evoked(
                    [d, s], weights=[1, -1]
                )
                # get raw ndarray for cluster-bas234ed stats as well
                raw_data = evoked[cond][f"mismatch_{i}"][p].get_data()
                evoked[cond][f"mismatch_{i}_data"][p] = raw_data

    return evoked


# %%
def extract_evokeds_2(epochs, participants):
    conds = ["harmonic", "inharmonic", "inharmonic_changing"]
    oddballs = ["deviant_1", "deviant_2", "deviant_3"]
    pitch_diffs = ["all", 50, 100, 150, 200, 250, 300]
    mismatches = [f"mismatch_{i}" for i in range(1, 4)]

    # this is to get rid of the pesky "NOTE" logs in the average() function
    mne.set_log_file("WARNING")
    evokeds = {}
    for c in conds:
        evokeds[c] = {}
        # do standards
        evokeds[c]["standard"] = {}
        for p in participants:
            query = f"pid == {p} and condition == '{c}' and stimulus == 'standard'"
            ev = epochs[query].average()
            evokeds[c]["standard"][p] = ev

        # do oddballs
        for o in oddballs:
            evokeds[c][o] = {}
            for p_diff in pitch_diffs:
                evokeds[c][o][p_diff] = {}
                if p_diff == "all":
                    pitch_diff_query = ""
                else:
                    pitch_diff_query = f" and pitch_diff == {p_diff} "

                for p in participants:
                    query = f"pid == {p} and condition == '{c}' and stimulus == '{o}' {pitch_diff_query}"
                    ev = epochs[query].average()
                    evokeds[c][o][p_diff][p] = ev

        # do mismatch responses (difference waves)
        for m in mismatches:
            evokeds[c][m] = {}
            for p_diff in pitch_diffs:
                evokeds[c][m][p_diff] = {}
                for p in participants:
                    s = evokeds[c]["standard"][p]
                    d = evokeds[c][f"deviant_{m[-1]}"][p_diff][p]
                    evokeds[c][m][p_diff][p] = mne.combine_evoked(
                        [d, s], weights=[1, -1]
                    )
    # set logging to default
    mne.set_log_file("INFO")
    return evokeds


def extract_orn(evokeds, participants):
    # calculate object-related-negativity
    orn = {
        "orn_standards": {"inharmonic": {}, "changing": {}},
        "orn_deviants": {"inharmonic": {}, "changing": {}},
    }
    for p in participants:
        # standards
        h = evokeds["harmonic"]["standard"][p]
        i = evokeds["inharmonic"]["standard"][p]
        c = evokeds["inharmonic_changing"]["standard"][p]
        orn["orn_standards"]["inharmonic"][p] = mne.combine_evoked(
            [i, h], weights=[1, -1]
        )
        orn["orn_standards"]["changing"][p] = mne.combine_evoked(
            [c, h], weights=[1, -1]
        )

        # deviants
        h = evokeds["harmonic"]["deviant_1"]["all"][p]
        i = evokeds["inharmonic"]["deviant_1"]["all"][p]
        c = evokeds["inharmonic_changing"]["deviant_1"]["all"][p]
        orn["orn_deviants"]["inharmonic"][p] = mne.combine_evoked(
            [i, h], weights=[1, -1]
        )
        orn["orn_deviants"]["changing"][p] = mne.combine_evoked([c, h], weights=[1, -1])

    return orn


def participant_peaks(evoked, mean_window=(-0.025, 0.025), plot=False, plot_fname=None):
    e = evoked.copy().pick(channel_picks)

    try:
        mmn_ch, mmn_lat, mmn_amp = e.get_peak(
            tmin=0.07, tmax=0.25, mode="neg", return_amplitude=True
        )

        # calculate MMN mean amplitude
        e_mean_crop = e.copy().crop(
            tmin=mmn_lat + mean_window[0], tmax=mmn_lat + mean_window[1]
        )
        mmn_mean_amp = e_mean_crop.data.mean() * 1e6
    except ValueError:
        mmn_ch, mmn_lat, mmn_amp, mmn_mean_amp = np.nan, np.nan, np.nan, np.nan

    # calculate mean amplitude in classical time window
    mmn_amp = e.copy().crop(0.1, 0.25).data.mean() * 1e6

    try:
        p3_ch, p3_lat, p3_amp = e.get_peak(
            tmin=0.15, tmax=0.4, mode="pos", return_amplitude=True
        )

        # calculate P3 mean amplitude
        e_mean_crop = e.copy().crop(
            tmin=p3_lat + mean_window[0], tmax=p3_lat + mean_window[1]
        )
        p3_mean_amp = e_mean_crop.data.mean() * 1e6
    except ValueError:
        p3_ch, p3_lat, p3_amp, p3_mean_amp = np.nan, np.nan, np.nan, np.nan

    if plot:
        fig, ax = plt.subplots()

        # plot evoked responses for each channel
        e.plot(picks=channel_picks, show=False, axes=ax)

        # mark MMN peak with a red dot
        ax.plot(mmn_lat, mmn_amp * 1e6, "ro")

        # mark P3 peak with a blue dot
        ax.plot(p3_lat, p3_amp * 1e6, "bo")

        fig.savefig(plots_path / "peak_checks" / plot_fname)

    ret = {
        "mmn_peak_ch": mmn_ch,
        "mmn_peak_lat": mmn_lat,
        "mmn_peak_amp": mmn_amp * 1e6,
        "mmn_mean_amp": mmn_mean_amp,
        "mmn_amp": mmn_amp,
        "p3_peak_ch": p3_ch,
        "p3_peak_lat": p3_lat,
        "p3_peak_amp": p3_amp * 1e6,
        "p3_mean_amp": p3_mean_amp,
    }

    return ret


def orn_peaks(evoked, mean_window=(-0.025, 0.025)):
    e = evoked.copy().pick(channel_picks)
    try:
        orn_ch, orn_lat, orn_amp = e.get_peak(
            tmin=0.1, tmax=0.3, mode="neg", return_amplitude=True
        )
        # calculate ORN mean amplitude
        e_mean_crop = e.copy().crop(
            tmin=orn_lat + mean_window[0], tmax=orn_lat + mean_window[1]
        )
        orn_mean_amp = e_mean_crop.data.mean() * 1e6
    except ValueError:
        orn_ch, orn_lat, orn_amp, orn_mean_amp = np.nan, np.nan, np.nan, np.nan

    ret = {
        "orn_peak_ch": orn_ch,
        "orn_peak_lat": orn_lat,
        "orn_peak_amp": orn_amp * 1e6,
        "orn_mean_amp": orn_mean_amp,
    }
    return ret


def p2_peaks(evoked, peak_time, mean_window=(-0.025, 0.025)):
    e = evoked.copy().pick(channel_picks)
    tmin, tmax = peak_time + mean_window[0], peak_time + mean_window[1]
    p2_amp = e.copy().crop(tmin, tmax).data.mean() * 1e6

    return p2_amp


# %%
# load epochs data
epochs = mne.read_epochs(epochs_path / "all_epochs-epo.fif", preload=True)

# %%
# extract evoked potentials
evoked = extract_evokeds_2(epochs, participants)

# %%
# extract orn
orn_evoked = extract_orn(evoked, participants)

# %%


# %%
conds = ["harmonic", "inharmonic", "inharmonic_changing"]
mismatches = [f"mismatch_{i}" for i in range(1, 4)]
pitch_diffs = ["all", 50, 100, 150, 200, 250, 300]


df_list = []

for c in conds:
    for m in mismatches:
        for p_diff in pitch_diffs:
            for p, e in evoked[c][m][p_diff].items():
                res = participant_peaks(e)
                res["pid"] = p
                res["condition"] = c
                res["mismatch"] = m
                res["pitch_diff"] = p_diff

                df_list.append(pd.DataFrame(res, index=[0]))

df = pd.concat(df_list, ignore_index=True)

# %%
df_orn_list = []

for ds in ["orn_deviants", "orn_standards"]:
    for c in ["inharmonic", "changing"]:
        for p in participants:
            e = orn_evoked[ds][c][p]
            res = orn_peaks(e)
            res["pid"] = p
            res["condition"] = c
            res["ds"] = ds
            df_orn_list.append(pd.DataFrame(res, index=[0]))

df_orn = pd.concat(df_orn_list, ignore_index=True)

# %%
# P2 peaks
# calculate MMN/ORN super-duper grand average and find latency
# Evokeds to include:
super_ga = [
    evoked["harmonic"]["mismatch_1"]["all"],
    evoked["inharmonic"]["mismatch_1"]["all"],
    evoked["inharmonic_changing"]["mismatch_1"]["all"],
    orn_evoked["orn_deviants"]["inharmonic"],
    orn_evoked["orn_deviants"]["changing"],
    orn_evoked["orn_standards"]["inharmonic"],
    orn_evoked["orn_standards"]["changing"],
]
# flatten
flat_ga = [i for e in super_ga for i in e.values()]

# grand average
ga = mne.grand_average(flat_ga)

# find peak
_, ga_peak = ga.copy().pick(channel_picks).get_peak(tmin=0.1, tmax=0.25, mode="neg")

print(f'Grand average peak for P2 calculations is {ga_peak} s.')

df_p2_list = []

for c in conds:
    for d in ["standard", "deviant_1"]:
        for p in participants:
            if d == "standard":
                e = evoked[c][d][p]
            else:
                e = evoked[c][d]["all"][p]
            res = {
                'p2_amp': p2_peaks(e, ga_peak, mean_window=(-0.04, 0.04)),
                "pid": p,
                "condition" : c,
                "deviance" : d,

            }
            df_p2_list.append(pd.DataFrame(res, index=[0]))

df_p2 = pd.concat(df_p2_list, ignore_index=True)


# %%
df.to_csv("../results/peak_measures.csv")
df_orn.to_csv("../results/peak_measures_orn.csv")
df_p2.to_csv("../results/peak_measures_p2.csv")

