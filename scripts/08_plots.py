# %%
# package imports
from importlib import reload
from matplotlib import pyplot as plt
import numpy as np
import mne
import pickle

from pyparsing import Combine
import shared_functions as sf

reload(sf)
# %%
# filepath to evokeds pickle
evokeds_fpath = "../results/evokeds/all.p"

# filepath to cluster pickle
clustering_path = "../results/clustering/"

# participants list
participants = list(range(1, 27)) + list(range(29, 38))

# filepath for plots
figpath = "../results/plots/"

# unpickle the evokeds
with open(evokeds_fpath, "rb") as f:
    evoked = pickle.load(f)

# unpickle cluster results
with open(clustering_path + "cluster_mismatches.p", "rb") as f:
    cluster_mismatches = pickle.load(f)

with open(clustering_path + "cluster_ftest.p", "rb") as f:
    cluster_ftest = pickle.load(f)

with open(clustering_path + "cluster_ttest.p", "rb") as f:
    cluster_ttest = pickle.load(f)

with open(clustering_path + "cluster_orn.p", "rb") as f:
    cluster_orn = pickle.load(f)

# get time vector for plotting clusters
time_vector = evoked['harmonic']['standard'][1].times

# %%
# STD_DEV_TOPOMAPS
fig, ax = plt.subplots(
    nrows=3, ncols=6, figsize=(6, 9), width_ratios=[5, 5, 5, 5, 5, 0.5]
)
plt.subplots_adjust(top=2, bottom=.2, hspace=1,)
topo_times = [0.12, 0.16, 0.2, 0.25, 0.4]
vlims = (-2, 2)

c = sf.get_evoked_list(evoked, "harmonic", "mismatch_1")
mne.grand_average(c).plot_topomap(
    times=topo_times, show=False, vlim=vlims, axes=ax[0][:]
)
c = sf.get_evoked_list(evoked, "inharmonic", "mismatch_1")
mne.grand_average(c).plot_topomap(
    times=topo_times, show=False, vlim=vlims, axes=ax[1][:]
)
c = sf.get_evoked_list(evoked, "inharmonic_changing", "mismatch_1")
mne.grand_average(c).plot_topomap(
    times=topo_times, show=False, vlim=vlims, axes=ax[2][:]
)
plt.savefig(f"{figpath}std_dev_topomaps.png", dpi=300,  bbox_inches='tight', pad_inches=0.5)


# STD_DEV_TRACES
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True)


# harmonic
comparison = {
    "Standard": sf.get_evoked_list(evoked, "harmonic", "standard"),
    "Deviant": sf.get_evoked_list(evoked, "harmonic", "deviant_1"),
    "Mismatch": sf.get_evoked_list(evoked, "harmonic", "mismatch_1"),
}
cluster_times = sf.identify_significant_clusters(cluster_mismatches['deviant_1']['harmonic'], time_vector)
sf.plot_evoked_comparison(comparison, ax=ax[0], clusters=cluster_times)


# inharmonic
comparison = {
    "Standard": sf.get_evoked_list(evoked, "inharmonic", "standard"),
    "Deviant": sf.get_evoked_list(evoked, "inharmonic", "deviant_1"),
    "Mismatch": sf.get_evoked_list(evoked, "inharmonic", "mismatch_1"),
}
cluster_times = sf.identify_significant_clusters(cluster_mismatches['deviant_1']['inharmonic'], time_vector)
sf.plot_evoked_comparison(comparison, ax=ax[1], show_sensors=False, show_legend=False, clusters=cluster_times)

comparison = {
    "Standard": sf.get_evoked_list(evoked, "inharmonic_changing", "standard"),
    "Deviant": sf.get_evoked_list(evoked, "inharmonic_changing", "deviant_1"),
    "Mismatch": sf.get_evoked_list(evoked, "inharmonic_changing", "mismatch_1"),
}
cluster_times = sf.identify_significant_clusters(cluster_mismatches['deviant_1']['inharmonic_changing'], time_vector)
sf.plot_evoked_comparison(comparison, ax=ax[2], show_sensors=False, show_legend=False, clusters=cluster_times)


# titles
ax[0].set_title("A) Harmonic", loc="left")
ax[1].set_title("B) Inharmonic", loc="left")
ax[2].set_title("C) Changing", loc="left")

plt.savefig(f"{figpath}std_dev_traces.png", dpi=300)

sf.stack_images(
    [
        f"{figpath}std_dev_traces.png",
        f"{figpath}std_dev_topomaps.png",
    ],
    f"{figpath}std_dev.png",
    orientation="horizontal",
)


# %%
# ERPS
comparison = {
    "harmonic": sf.get_evoked_list(evoked, "harmonic", "mismatch_1"),
    "inharmonic": sf.get_evoked_list(evoked, "inharmonic", "mismatch_1"),
    "changing": sf.get_evoked_list(evoked, "inharmonic_changing", "mismatch_1"),
}

sf.plot_evoked_comparison(comparison, ylim={"eeg": (-2, 4)})
plt.savefig(f"{figpath}erps.png", dpi=300)
plt.show()


# %%
# MISMATCHES ERPS

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 12))
ylim = dict(eeg=(-2, 2.5))
c = {
    "harmonic": sf.get_evoked_list(evoked, "harmonic", "mismatch_1"),
    "inharmonic": sf.get_evoked_list(evoked, "inharmonic", "mismatch_1"),
    "changing": sf.get_evoked_list(evoked, "inharmonic_changing", "mismatch_1"),
}
sf.plot_evoked_comparison(c, ax=ax[0], ylim=ylim, show_legend=True)

c = {
    "harmonic": sf.get_evoked_list(evoked, "harmonic", "mismatch_2"),
    "inharmonic": sf.get_evoked_list(evoked, "inharmonic", "mismatch_2"),
    "changing": sf.get_evoked_list(evoked, "inharmonic_changing", "mismatch_2"),
}
sf.plot_evoked_comparison(c, ax=ax[1], ylim=ylim, show_legend=False)

c = {
    "harmonic": sf.get_evoked_list(evoked, "harmonic", "mismatch_3"),
    "inharmonic": sf.get_evoked_list(evoked, "inharmonic", "mismatch_3"),
    "changing": sf.get_evoked_list(evoked, "inharmonic_changing", "mismatch_3"),
}
sf.plot_evoked_comparison(c, ax=ax[2], ylim=ylim, show_legend=False)
# titles
ax[0].set_title("A) Mismatch #1", loc="left")
ax[1].set_title("B) Mismatch #2", loc="left")
ax[2].set_title("C) Mismatch #3", loc="left")

plt.savefig(f"{figpath}mismatches_erps.png", dpi=300)



# %%
fig, axs = plt.subplots(3, 3, figsize=(20, 15))

conds = ["harmonic", "inharmonic", "inharmonic_changing"]
deviants = [f"deviant_{i}" for i in range(1, 4)]

# for pretty printout
condition_names = ["Harmonic", "Inharmonic", "Changing"]
dev_names = ["deviant #1", "deviant #2", "deviant #3"]

for i, c in enumerate(conds):
    for j, d in enumerate(deviants):
        comparison = {
            "standard": sf.get_evoked_list(evoked, c, "standard"),
            "deviant": sf.get_evoked_list(evoked, c, d),
            "mismatch": sf.get_evoked_list(evoked, c, f"mismatch_{j+1}"),
        }
        sf.plot_evoked_comparison(comparison, ax=axs[i][j])

        axs[i][j].set_title(f"{condition_names[i]}, {dev_names[j]}")

        # sig_clusters = get_sig_cluster_times(cr[d][c])
        # for clust in sig_clusters:
        # draw_sig_rectangle(clust["t_start"], clust["t_stop"], axs[i][j])

plt.savefig("../results/plots/mismatches_std_dev.png", dpi=300)
plt.show()
# %%
# CLUSTER PLOTS
ylims = (-2.3, 4.5)
# F-test results
e = {
    "harmonic": list(evoked["harmonic"]["mismatch_1"].values()),
    "inharmonic": list(evoked["inharmonic"]["mismatch_1"].values()),
    "changing": list(evoked["inharmonic_changing"]["mismatch_1"].values()),
}

sf.plot_clusters(
    e,
    cluster_ftest["mismatch_1"],
    ylims=ylims,
    title="A) F-test for differences between conditions",
    topo_vlim=(0,15),
    map_colors='Reds'
)
sf.save_plot("cluster_ftest.png")

# %%
# t-test contrasts
# harmonic vs inharmonic
e = {
    "harmonic": list(evoked["harmonic"]["mismatch_1"].values()),
    "inharmonic": list(evoked["inharmonic"]["mismatch_1"].values()),
}
sf.plot_clusters(
    e,
    cluster_ttest["mismatch_1"]["harm/ih"],
    ylims=ylims,
    p_cutoff=0.05 / 3,
    title="B) Post-hoc t-test: harmonic vs inharmonic",
    topo_vlim=(-8,8),
    map_colors='RdBu_r'
)
sf.save_plot("cluster_contrast_harm_ih.png")

# %% harmonic vs changing
e = {
    "harmonic": list(evoked["harmonic"]["mismatch_1"].values()),
    "changing": list(evoked["inharmonic_changing"]["mismatch_1"].values()),
}
sf.plot_clusters(
    e,
    cluster_ttest["mismatch_1"]["harm/ic"],
    ylims=ylims,
    p_cutoff=0.05 / 3,
    title="C) Post-hoc t-test: harmonic vs changing",
    topo_vlim=(-8,8),
    map_colors='RdBu_r'

)
sf.save_plot("cluster_contrast_harm_changing.png")

# %% inharmonic vs changing
e = {
    "inharmonic": list(evoked["inharmonic"]["mismatch_1"].values()),
    "changing": list(evoked["inharmonic_changing"]["mismatch_1"].values()),
}
sf.plot_clusters(
    e,
    cluster_ttest["mismatch_1"]["ih/ic"],
    ylims=ylims,
    p_cutoff=0.05 / 3,
    title="D) Post-hoc t-test: inharmonic vs changing",
    topo_vlim=(-8,8),
    map_colors='RdBu_r'

)
sf.save_plot("cluster_contrast_inharm_changing.png")

# %%
# Combine all
c = [
    "cluster_ftest.png",
    "cluster_contrast_harm_ih.png",
    "cluster_contrast_harm_changing.png",
    "cluster_contrast_inharm_changing.png",
]
c = [f"{figpath}{i}" for i in c]

sf.stack_images(c, f"{figpath}clusters.png", padding="right")

# %%
# ORN
ylims = (-3.2, 3)
# ORN inharmonic standard
# # %%
e = {
    "harmonic standard": list(evoked["harmonic"]["standard"].values()),
    "inharmonic standard": list(evoked["inharmonic"]["standard"].values()),
    "ORN": list(evoked["orn_standards"]["inharmonic"].values()),
}
sf.plot_clusters(
    e,
    cluster_orn["standard"]["inharmonic"],
    ylims=ylims,
    title="A) ORN analysis: harmonic vs inharmonic standards",
    topo_vlim=(-10,10),
    map_colors='RdBu_r'
)
sf.save_plot("orn_inharmonic_standard.png")
plt.show()
# %%
# ORN inharmonic deviant
e = {
    "harmonic deviant": list(evoked["harmonic"]["deviant_1"].values()),
    "inharmonic deviant": list(evoked["inharmonic"]["deviant_1"].values()),
    "ORN": list(evoked["orn_deviants"]["inharmonic"].values()),
}
sf.plot_clusters(
    e,
    cluster_orn["deviant_1"]["inharmonic"],
    ylims=ylims,
    title="B) ORN analysis: harmonic vs inharmonic deviants",
    topo_vlim=(-10,10),
    map_colors='RdBu_r'

)
sf.save_plot("orn_inharmonic_deviant.png")
plt.show()
# %%
# ORN Changing standard
e = {
    "harmonic standard": list(evoked["harmonic"]["standard"].values()),
    "changing standard": list(evoked["inharmonic_changing"]["standard"].values()),
    "ORN": list(evoked["orn_standards"]["changing"].values()),
}

sf.plot_clusters(
    e,
    cluster_orn["standard"]["changing"],
    ylims=ylims,
    title="C) ORN analysis: harmonic vs changing standards",
    topo_vlim=(-10,10),
    map_colors='RdBu_r'

)
sf.save_plot("orn_changing_standard.png")


# %%
# ORN Changing deviant
e = {
    "harmonic deviant": list(evoked["harmonic"]["deviant_1"].values()),
    "changing deviant": list(evoked["inharmonic_changing"]["deviant_1"].values()),
    "ORN": list(evoked["orn_deviants"]["changing"].values()),
}
sf.plot_clusters(
    e,
    cluster_orn["deviant_1"]["changing"],
    ylims=ylims,
    title="D) ORN analysis: harmonic vs changing deviants",
    topo_vlim=(-10,10),
    map_colors='RdBu_r'

)
sf.save_plot(f"orn_changing_deviant.png")


# %% ORN Contrast standards
e = {
    "ORN inharmonic": list(evoked["orn_standards"]["inharmonic"].values()),
    "ORN changing": list(evoked["orn_standards"]["changing"].values()),
}
sf.plot_clusters(
    e,
    cluster_orn["standard"]["contrast"],
    ylims=ylims,
    title="E) ORN contrast: inharmonic vs changing standards",
    topo_vlim=(-10,10),
    map_colors='RdBu_r'

)
sf.save_plot(f"orn_contrast_standard.png")

# %% ORN Contrast deviants
e = {
    "ORN inharmonic": list(evoked["orn_deviants"]["inharmonic"].values()),
    "ORN changing": list(evoked["orn_deviants"]["changing"].values()),
}
sf.plot_clusters(
    e,
    cluster_orn["deviant_1"]["contrast"],
    ylims=ylims,
    title="F) ORN contrast: inharmonic vs changing deviants",
    topo_vlim=(-10,10),
    map_colors='RdBu_r'

)

sf.save_plot(f"orn_contrast_deviant.png")

# %%
# Combine
c = [
    "orn_inharmonic_standard.png",
    "orn_inharmonic_deviant.png",
    "orn_changing_standard.png",
    "orn_changing_deviant.png",
    # "orn_contrast_standard.png",
    # "orn_contrast_deviant.png",
]
c = [f"{figpath}{i}" for i in c]
sf.stack_images(c, f"{figpath}orn.png", padding="right")

