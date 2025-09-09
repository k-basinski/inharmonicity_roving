# %%
import pathlib
import mne
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shared_functions as sf
# %%
# filepath to evokeds pickle
evokeds_fpath = "../results/evokeds/all.p"

# participants list
participants = list(range(1, 27)) + list(range(29, 38))

# channel picks for plotting
channel_picks = ["F3", "Fz", "F4", "FC1", "FC2"]

# raw data filepath for channel adjacency
raw_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/raw")

# filepath for plots
figpath = "../results/plots/"

# unpickle the evokeds
with open(evokeds_fpath, "rb") as f:
    evokeds = pickle.load(f)
# %%
# ORN for inharmonic
evokeds["orn"] = {
    "inharmonic_standard": {},
    "changing_standard": {},
    "inharmonic_deviant": {},
    "changing_deviant": {},
}
for p in participants:
    h = evokeds["harmonic"]["standard"][p]
    i = evokeds["inharmonic"]["standard"][p]
    c = evokeds["inharmonic_changing"]["standard"][p]
    evokeds["orn"]["inharmonic_standard"][p] = mne.combine_evoked(
        [i, h], weights=[1, -1]
    )
    evokeds["orn"]["changing_standard"][p] = mne.combine_evoked([c, h], weights=[1, -1])
    h = evokeds["harmonic"]["deviant_1"][p]
    i = evokeds["inharmonic"]["deviant_1"][p]
    c = evokeds["inharmonic_changing"]["deviant_1"][p]
    evokeds["orn"]["inharmonic_deviant"][p] = mne.combine_evoked(
        [i, h], weights=[1, -1]
    )
    evokeds["orn"]["changing_deviant"][p] = mne.combine_evoked([c, h], weights=[1, -1])
# %%
comp = {
    "harmonic standard": list(evokeds["harmonic"]["standard"].values()),
    "inharmonic standard": list(evokeds["inharmonic"]["standard"].values()),
    "difference (ORN)": list(evokeds["orn"]["inharmonic_standard"].values()),
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine="mean")
# %%
comp = {
    "harmonic standard": list(evokeds["harmonic"]["standard"].values()),
    "changing standard": list(evokeds["inharmonic_changing"]["standard"].values()),
    "difference (ORN)": list(evokeds["orn"]["changing_standard"].values()),
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine="mean")
# %%
comp = {
    "harmonic deviant": list(evokeds["harmonic"]["deviant_1"].values()),
    "inharmonic deviant": list(evokeds["inharmonic"]["deviant_1"].values()),
    "difference (ORN)": list(evokeds["orn"]["inharmonic_deviant"].values()),
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine="mean")
# %%
comp = {
    "harmonic deviant": list(evokeds["harmonic"]["deviant_1"].values()),
    "changing deviant": list(evokeds["inharmonic_changing"]["deviant_1"].values()),
    "difference (ORN)": list(evokeds["orn"]["changing_deviant"].values()),
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine="mean")
# %%
comp = {
    "standard harmonic vs inharmonic": list(
        evokeds["orn"]["inharmonic_standard"].values()
    ),
    "standard harmonic vs changing": list(evokeds["orn"]["changing_standard"].values()),
    "deviant harmonic vs inharmonic": list(
        evokeds["orn"]["inharmonic_deviant"].values()
    ),
    "deviant harmonic vs changing": list(evokeds["orn"]["changing_deviant"].values()),
}

mne.viz.plot_compare_evokeds(comp, picks=channel_picks, combine="mean")

# %%
# compare ORN topography against MMN
mmn_ga = mne.grand_average(list(evokeds["harmonic"]["mismatch_1"].values()))
orn_ga = mne.grand_average(list(evokeds["orn"]["inharmonic_standard"].values()))
# %%
topo_times = np.arange(0.1, 0.22, 0.02)
vlims = (-1.8, 1.8)

fig = mmn_ga.plot_topomap(times=topo_times, vlim=vlims, show=False)
fig.suptitle('MMN topographies')
sf.save_plot('sf3a.png')

fig = orn_ga.plot_topomap(times=topo_times, vlim=vlims, show=False);
fig.suptitle('ORN topographies')
sf.save_plot('sf3b.png')

# Combine
c = [
    "sf3a.png",
    "sf3b.png",
]
c = [f"{figpath}{i}" for i in c]
sf.stack_images(c, f"{figpath}sf3.png", padding="right")
# %%
def cluster_compare_mmn_orn(mmn_source, orn_source):
    # cluster_based permutations - MMN vs ORN
    t_window = (.1, .2)
    # prepare data
    a = np.array([i.crop(*t_window).get_data() for i in evokeds[mmn_source]['mismatch_1'].values()])
    b = np.array([i.crop(*t_window).get_data() for i in evokeds['orn'][orn_source].values()])

    # transpose so it fits the shape
    a_t = np.transpose(a, (0, 2, 1))
    b_t = np.transpose(b, (0, 2, 1))

    # cluster permutations
    cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(
        a_t - b_t,
        n_permutations=1024*8,
        tail=0,
        threshold=None,
        n_jobs=-1,
        adjacency=adjacency,
        out_type="mask",
    )
    return cluster_stats

mmn_sources = ['harmonic', 'inharmonic']
orn_sources = ['inharmonic_standard', 'inharmonic_deviant', 'changing_standard']

i = 1
for ms in mmn_sources:
    for os in orn_sources:
        res  = cluster_compare_mmn_orn(ms, os)
        print(f'{i}) Comparing {ms} - {os}: {np.where(res[2] < .05)}')
        i += 1
# %%
