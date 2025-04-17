# %%
# package imports
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import mne
import pickle
# %%

# config

# project file path
# project_path = pathlib.Path(__file__).parents[1]
project_path = pathlib.Path("/Users/kbas/cloud/sci/harmonicity_roving")

# plots patch
plots_path = project_path / "results" / "plots"

# path for storing clustering results
clustering_path = project_path / "results" / "clustering"

# epochs filepath
epochs_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/epochs")

# raw data filepath for channel adjacency
raw_path = pathlib.Path("/Users/kbas/sci_data/harmonicity_data/raw")

# participants list
participants = list(range(1, 27)) + list(range(29, 38))

# channel picks for plotting
channel_picks = ["F3", "Fz", "F4", "FC1", "FC2"]


# %%
def traverse_dict(d: dict, indent: str = ""):
    """Print keys of a (nested) dictionary.

    Args:
        d (dict): A dict to traverse
        indent (str, optional): Starting indent character. Defaults to "".
    """
    for k, v in d.items():
        print(indent, k)
        if isinstance(v, dict):
            traverse_dict(v, indent + "-")


def read_epochs(epochs_path, participants):
    epochs_list = []
    # iterate over all participants
    for p in participants:
        # form path to epochs
        fpath = epochs_path / f"{p}" / f"{p}-epo.fif"
        e = mne.read_epochs(fpath, preload=True)
        meta, _, _ = mne.epochs.make_metadata(
            e.events, e.event_id, tmin=-0.1, tmax=0.5, sfreq=e.info["sfreq"]
        )
        meta["pid"] = p
        e.metadata = meta
        epochs_list.append(e)

    epochs = mne.concatenate_epochs(epochs_list)

    # edit metadata so it's usable
    m = epochs.metadata.loc[:, ["event_name", "pid"]]
    m_ex = m.event_name.str.split("/", expand=True)
    m["stimulus"] = m_ex[0]
    m["condition"] = m_ex[1]
    m["pitch_diff"] = pd.to_numeric(m_ex[2])
    epochs.metadata = m

    # return epochs object
    return epochs


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
        "orn_standards": {"inharmonic": {}, "changing": {}},
        "orn_deviants": {"inharmonic": {}, "changing": {}},
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
                # get raw ndarray for cluster-based stats as well
                raw_data = evoked[cond][f"mismatch_{i}"][p].get_data()
                evoked[cond][f"mismatch_{i}_data"][p] = raw_data

    # calculate object-related-negativity
    for p in participants:
        # standards
        h = evoked["harmonic"]["standard"][p]
        i = evoked["inharmonic"]["standard"][p]
        c = evoked["inharmonic_changing"]["standard"][p]
        evoked["orn_standards"]["inharmonic"][p] = mne.combine_evoked(
            [i, h], weights=[1, -1]
        )
        evoked["orn_standards"]["changing"][p] = mne.combine_evoked(
            [c, h], weights=[1, -1]
        )
        
        # deviants
        h = evoked["harmonic"]["deviant_1"][p]
        i = evoked["inharmonic"]["deviant_1"][p]
        c = evoked["inharmonic_changing"]["deviant_1"][p]
        evoked["orn_deviants"]["inharmonic"][p] = mne.combine_evoked(
            [i, h], weights=[1, -1]
        )
        evoked["orn_deviants"]["changing"][p] = mne.combine_evoked(
            [c, h], weights=[1, -1]
        )
    return evoked


def evoked_ndarrays(evoked, conds):
    evoked_arrays = {}
    for cond in conds:
        evoked_arrays[cond] = {}
        mismatches = [f"mismatch_{i}" for i in range(1, 4)]

        # for every type of mismatch...
        for mismatch in mismatches:
            # extract values to ndarray
            res = np.array(list(evoked[cond][f"{mismatch}_data"].values()))

            # transpose to fit cluster function requirements
            res_t = np.transpose(res, (0, 2, 1))

            # write out to evoked_arrays dict
            evoked_arrays[cond][mismatch] = res_t

        # ...and for standard...
        res = np.array(list(evoked[cond]["standard_data"].values()))

        # transpose to fit cluster function requirements
        res_t = np.transpose(res, (0, 2, 1))

        evoked_arrays[cond]["standard"] = res_t

        # ... and for every type of deviant.
        deviants = [f"deviant_{i}" for i in range(1, 4)]
        for d in deviants:
            # extract values to ndarray
            res = np.array(list(evoked[cond][f"{d}_data"].values()))

            # transpose to fit cluster function requirements
            res_t = np.transpose(res, (0, 2, 1))

            # write out to evoked_arrays dict
            evoked_arrays[cond][d] = res_t

    return evoked_arrays


def get_evoked_list(evoked: dict, cond: str, odd: str) -> list:
    """Returns a list of evoked objects for a condition and oddball, useful
    for plotting.

    Args:
        evoked (dict): Dict with evoked objects
        cond (str): Condition
        odd (str): Oddball

    Returns:
        list: List of evoked objects
    """
    res = [e for e in evoked[cond][odd].values()]
    return res


def get_adjacency_matrix(raw_path: str = None, plot=False):
    if raw_path is None:
        # read build-in MNE adjacency for Easycap 64
        adjacency, ch_names = mne.channels.read_ch_adjacency("easycap64ch-avg")
    else:
        # load up one raw object to get adjacency
        raw = mne.io.read_raw_brainvision(
            raw_path,
            eog=["CanEye", "LowEye"],
            misc=["Heart"],
        )

        # read adjacency matrix from montage
        raw.set_montage("standard_1020")
        adjacency, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type="eeg")

        if plot:
            mne.viz.plot_ch_adjacency(raw.info, adjacency, ch_names)

    return adjacency, ch_names


def cluster_permutations(a, b, adjacency, n_perm=1024, threshold_tfce=None):
    cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(
        a - b,
        n_permutations=n_perm,
        tail=0,
        threshold=threshold_tfce,
        n_jobs=-1,
        adjacency=adjacency,
        out_type="mask",
    )
    return cluster_stats


def calculate_cluster_contrasts(evoked_arrays, adjacency):
    mismatches = [f"mismatch_{i}" for i in range(1, 4)]
    cluster_results = {}
    for m in mismatches:
        cluster_results[m] = {
            "harm/ih": cluster_permutations(
                evoked_arrays["harmonic"][m],
                evoked_arrays["inharmonic"][m],
                adjacency,
                n_perm=1e4,
            ),
            "ih/ic": cluster_permutations(
                evoked_arrays["inharmonic"][m],
                evoked_arrays["inharmonic_changing"][m],
                adjacency,
                n_perm=1e4,
            ),
            "harm/ic": cluster_permutations(
                evoked_arrays["harmonic"][m],
                evoked_arrays["inharmonic_changing"][m],
                adjacency,
                n_perm=1e4,
            ),
        }

    return cluster_results


def calculate_clusters_ftest(X, adjacency, thresh=None, n_permutations=1024):
    cluster_stats = mne.stats.spatio_temporal_cluster_test(
        X,
        n_permutations=n_permutations,
        threshold=thresh,
        tail=1,  # We are running an F test, so we look at the upper tail
        n_jobs=-1,  # use all cores
        buffer_size=None,
        seed=666,
        adjacency=adjacency,
        out_type="mask",
    )

    return cluster_stats


def calculate_cluster_mismatches(evoked_arrays, adjacency):
    deviants = [f"deviant_{i}" for i in range(1, 4)]
    cluster_results = {}
    for d in deviants:
        cluster_results[d] = {
            "harmonic": cluster_permutations(
                evoked_arrays["harmonic"]["standard"],
                evoked_arrays["harmonic"][d],
                adjacency,
                n_perm=1e4,
                # threshold_tfce=tfce,
            ),
            "inharmonic": cluster_permutations(
                evoked_arrays["inharmonic"]["standard"],
                evoked_arrays["inharmonic"][d],
                adjacency,
                n_perm=1e4,
                # threshold_tfce=tfce,
            ),
            "inharmonic_changing": cluster_permutations(
                evoked_arrays["inharmonic_changing"]["standard"],
                evoked_arrays["inharmonic_changing"][d],
                adjacency,
                n_perm=1e4,
                # threshold_tfce=tfce,
            ),
        }

    return cluster_results


def calculate_cluster_orn(evoked_arrays, adjacency):
    orn_results = {}
    for d in ['standard', 'deviant_1']:
        orn_results[d] = {
            "inharmonic": cluster_permutations(
                evoked_arrays["harmonic"][d],
                evoked_arrays["inharmonic"][d],
                adjacency,
                n_perm=1e4,
            ),
            "changing": cluster_permutations(
                evoked_arrays["harmonic"][d],
                evoked_arrays["inharmonic_changing"][d],
                adjacency,
                n_perm=1e4,
            ),
            "contrast": cluster_permutations(
                evoked_arrays["inharmonic"][d] - evoked_arrays["harmonic"][d],
                evoked_arrays["inharmonic_changing"][d] - evoked_arrays["harmonic"][d],
                adjacency,
                n_perm=1e4,
            )
        }

    return orn_results


def print_sig_clusters(cluster_stats, p_thresh=0.05):
    F_obs, clusters, pvals, _ = cluster_stats
    samples_in_epoch = 551
    # make times vector
    times = (np.arange(samples_in_epoch) * 0.001) - 0.1

    for c in range(len(clusters)):
        if pvals[c] < p_thresh:
            print(f"Cluster {c}, p = {pvals[c]}")
            t_start = times[np.any(clusters[c], axis=1)][0]
            t_stop = times[np.any(clusters[c], axis=1)][-1]
            no_sensors = np.max(clusters[c].sum(axis=1))
            print(f"{round(t_start, 3)} - {round(t_stop, 3)}")
            print(f"Sensors {no_sensors}/30")
            print()


def print_all_clusters(cluster_stats):
    F_obs, clusters, pvals, _ = cluster_stats
    samples_in_epoch = 551
    # make times vector
    times = (np.arange(samples_in_epoch) * 0.001) - 0.1
    cluster_list = []
    for c in range(len(clusters)):
        t_start = times[np.any(clusters[c], axis=1)][0]
        t_stop = times[np.any(clusters[c], axis=1)][-1]
        no_sensors = np.max(clusters[c].sum(axis=1))
        cluster_data = {
            'cluster_id': c,
            'p-value': pvals[c],
            't_start': round(t_start, 3),
            't_stop': round(t_stop, 3),
            'sensors': no_sensors,
        }
        cluster_list.append(cluster_data)
    
    return pd.DataFrame(cluster_list)



def get_sig_cluster_times(cluster_stats, p_thresh=0.05):
    _, clusters, pvals, _ = cluster_stats
    # make times vector
    times = (np.arange(551) * 0.001) - 0.1

    sig_clusters = []

    for c in range(len(clusters)):
        if pvals[c] < p_thresh:
            print(f"Cluster {c}, p = {pvals[c]}")
            t_start = times[np.any(clusters[c], axis=1)][0]
            t_stop = times[np.any(clusters[c], axis=1)][-1]
            no_sensors = np.max(clusters[c].sum(axis=1))
            print(f"{round(t_start, 3)} - {round(t_stop, 3)}")
            print(f"Sensors: {no_sensors}/32")
            sig_cluster = {
                "cluster#": c,
                "pval": pvals[c],
                "t_start": t_start,
                "t_stop": t_stop,
                "no_sensors": f"{no_sensors}/32",
            }
            sig_clusters.append(sig_cluster)

    return sig_clusters


def plot_individual_evoked(participants, evoked):
    for p in participants:
        ind_e = evoked["harmonic"]["deviant_1"][p]
        title = {"eeg": f"Participant {p}, EEG"}
        ind_e.plot(ylim={"eeg": (-5, 5)}, titles=title)


def default_colors(conditions):
    conds = ["harmonic", "inharmonic", "inharmonic_changing"]
    colors = ["crimson", "tab:blue", "tab:olive"]
    colors_dict = dict(zip(conds, colors))
    ret = {c: colors_dict[c] for c in conditions}
    return ret


def save_plot(
    prefix: str, conditions: list, mismatch: str, cluster_id: int, dpi: int = 300
):
    """Saves the plot with a formatted prefix and specified dpi.

    Args:
        prefix (str): Prefix, first part of filename
        conditions (list): List of conditions being contrasted
        mismatch (str): Mismatch number
        cluster_id (int): Cluster number
        dpi (int, optional): Plot DPI. Defaults to 300.
    """
    # define contrast abbreviations
    condition_abbrevs = {
        "harmonic": "harm",
        "inharmonic": "ih",
        "inharmonic_changing": "ic",
    }
    contrast_label = f"{prefix}_m{mismatch[-1]}"
    for cond in conditions:
        contrast_label += "-"
        contrast_label += condition_abbrevs[cond]
    contrast_label += f"_c{cluster_id}.png"
    try:
        plt.savefig(plots_path / contrast_label, dpi=dpi)
    except Exception:
        print("Error saving plot")
    else:
        print(f"Plot saved successfully at {contrast_label}")


def plot_ftest_clusters(evoked, clustering_results, mismatch, cluster_id=0):
    F_obs, clusters, p_values, h0 = clustering_results
    p_cutoff = 0.05
    conditions = ["harmonic", "inharmonic", "inharmonic_changing"]

    # pick one cluster to plot
    good_clusters = np.where(p_values < p_cutoff)[0]

    try:
        cluster = [clusters[g] for g in good_clusters][cluster_id]
    except IndexError:
        print(f"No cluster by that id, cluster count is {len(good_clusters)}")
        return None

    # get the info object
    evokeds_info = evoked["harmonic"][mismatch][1].info

    # get topography for F stat
    # average over the timecourse of the cluster
    f_map = (F_obs * cluster).mean(axis=0)

    # create spatial mask
    mask = (cluster.sum(axis=0) > 0)[:, np.newaxis]

    # create time mask
    t_mask = cluster.sum(axis=1) > 0

    comparison = {k: get_evoked_list(evoked, k, mismatch) for k in conditions}

    # get cluster times
    sig_times = evoked["harmonic"][mismatch][1].times[t_mask]

    ch_inds = np.squeeze(mask).nonzero()[0]
    colors = default_colors(conditions)

    # Format title
    combine = "gfp"
    title = f"Cluster #{cluster_id}, {len(ch_inds)} sensors"

    fig, ax = plt.subplots(
        nrows=1, ncols=3, gridspec_kw={"width_ratios": (10, 1, 50)}, figsize=(12, 4)
    )

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], evokeds_info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        mask=mask,
        cmap="Reds",
        vlim=(np.min, np.max),
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
        axes=ax[0],
    )

    # remove the title that would otherwise say "0.000 s"
    ax[0].set_title("")

    # add axes for colorbar
    plt.colorbar(ax[0].images[0], cax=ax[1])
    ax[0].set_xlabel(
        "Averaged F-map \n ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
    )

    # plot evoked GFP plot
    mne.viz.plot_compare_evokeds(
        comparison,
        title=title,
        picks=ch_inds,
        axes=ax[2],
        colors=colors,
        show=False,
        split_legend=True,
        truncate_yaxis="auto",
        combine=combine,
    )

    # plot temporal cluster extent
    ymin, ymax = ax[2].get_ylim()
    ax[2].fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color="tab:grey", alpha=0.2
    )

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)
    save_plot("ftest", [], mismatch, cluster_id)
    plt.show()


def plot_ttest_clusters(
    evoked,
    clustering_results,
    conditions,
    mismatch="mismatch_1",
    cluster_id=0,
    combine="gfp",
):
    t_obs, clusters, p_values, h0 = clustering_results
    p_cutoff = 0.05

    # pick one cluster to plot
    good_clusters = np.where(p_values < p_cutoff)[0]

    try:
        cluster = [clusters[g] for g in good_clusters][cluster_id]
    except IndexError:
        print(f"No cluster by that id, cluster count is {len(good_clusters)}")
        return None

    # get the info object
    evokeds_info = evoked["harmonic"][mismatch][1].info

    # get topography for F stat
    # average over the timecourse of the cluster
    t_map = (t_obs * cluster).mean(axis=0)

    # scale by 10^-6 so that units are the same
    # this is a hacky solution, beware
    # t_map = t_map * 10e-6

    # create spatial mask
    mask = (cluster.sum(axis=0) > 0)[:, np.newaxis]

    # create time mask
    t_mask = cluster.sum(axis=1) > 0

    comparison = {k: get_evoked_list(evoked, k, mismatch) for k in conditions}

    colors = default_colors(conditions)

    # get cluster times
    sig_times = evoked["harmonic"][mismatch][1].times[t_mask]

    if combine == "gfp":
        ch_inds = np.squeeze(mask).nonzero()[0]
    else:
        ch_inds = channel_picks

    # Format title
    title = f"Cluster #{cluster_id}, {len(ch_inds)} sensors"

    fig, ax = plt.subplots(
        nrows=1, ncols=2, gridspec_kw={"width_ratios": (10, 50)}, figsize=(10, 3.5)
    )

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(t_map[:, np.newaxis], evokeds_info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        mask=mask,
        cmap="Reds",
        vlim=(np.min, np.max),
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
        axes=ax[0],
    )

    # remove the title that would otherwise say "0.000 s"
    ax[0].set_title("")

    # add axes for colorbar
    plt.colorbar(ax[0].images[0], ax=ax[0])
    ax[0].set_xlabel(
        "Averaged t-map \n ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
    )

    # plot evoked plot
    mne.viz.plot_compare_evokeds(
        comparison,
        title=title,
        picks=ch_inds,
        axes=ax[1],
        colors=colors,
        show=False,
        split_legend=True,
        truncate_yaxis="auto",
        combine=combine,
        show_sensors=True,
    )

    # plot temporal cluster extent
    ymin, ymax = ax[1].get_ylim()
    ax[1].fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color="tab:grey", alpha=0.2
    )

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)

    # save file
    # format filename
    save_plot("ttest", conditions, mismatch=mismatch, cluster_id=cluster_id)
    plt.show()


def plot_ttest_clusters2(
    evokeds, clustering_results, title=None, cluster_id=0, p_cutoff=0.05
):
    """Plots clustering results from cluster-based permutations (t-test comparing two conditions).

    Args:
        evokeds (dict): A dict of lists of evoked responses. Dict keys determine plot labels.
        clustering_results (tuple): Results of cluster-based permutations. A tuple contains t_obs, clusters, p_values, h0. This can taken from output of mne.stats.spatio_temporal_cluster_1samp_test.
        cluster_id (int, optional): Id of the cluster to plot. This is an id from a list of consecutive, statistically significant clusters. Defaults to 0.
        p_cutoff (float, optional): P-value to cutoff significant clusters. Defaults to 0.

    Returns:
        fig: matplotlib.figure.Figure
    """

    from matplotlib import ticker

    # unpack from clustering results
    t_obs, clusters, p_values, _ = clustering_results

    # pick one cluster to plot
    good_clusters = np.where(p_values < p_cutoff)[0]

    try:
        cluster = [clusters[g] for g in good_clusters][cluster_id]
    except IndexError:
        print(f"No cluster by that id, cluster count is {len(good_clusters)}")
        return None

    # get the cluster p-value for plotting
    pv = p_values[good_clusters[cluster_id]]

    # get the info object from the first dict element
    conditions = list(evokeds.keys())
    evokeds_info = evokeds[conditions[0]][0].info

    # get topography for F stat
    # average over the timecourse of the cluster
    t_map = (t_obs * cluster).mean(axis=0)

    # scale by 10^-6 so that units are the same
    # this is a hacky solution, beware
    # t_map = t_map * 10e-6

    # create spatial mask
    mask = (cluster.sum(axis=0) > 0)[:, np.newaxis]

    # create time mask
    t_mask = cluster.sum(axis=1) > 0

    # colors = default_colors(conditions)

    # get cluster times
    sig_times = evokeds[conditions[0]][1].times[t_mask]

    ch_inds = np.squeeze(mask).nonzero()[0]

    combine = "gfp"

    # Format title
    if title is None:
        title = f"Cluster #{cluster_id}, {len(ch_inds)} sensors"

    fig, ax = plt.subplots(
        nrows=1, ncols=2, gridspec_kw={"width_ratios": (10, 50)}, figsize=(12, 4)
    )

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(t_map[:, np.newaxis], evokeds_info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        mask=mask,
        cmap="Reds",
        vlim=(np.min, np.max),
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
        axes=ax[0],
    )

    # remove the title that would otherwise say "0.000 s"
    ax[0].set_title("")

    # add axes for colorbar
    plt.colorbar(
        ax[0].images[0],
        ax=ax[0],
        format=ticker.FuncFormatter(lambda x, pos: round(x / 100000)),
    )
    # ax[0].set_xlabel(
    #     "Averaged t-map \n ({:0.3f} - {:0.3f} s) \n p = {}".format(sig_times[[0]], sig_times[[-1]], pv)
    # )

    ax[0].set_xlabel(
        f"Averaged t-map \n ({sig_times[0]:.3f} - {sig_times[-1]:.3f} s) \np = {pv:.3f}"
    )
    # plot evoked GFP plot
    mne.viz.plot_compare_evokeds(
        evokeds,
        title=title,
        picks=ch_inds,
        axes=ax[1],
        # colors=colors,
        show=False,
        split_legend=True,
        truncate_yaxis="auto",
        combine=combine,
    )

    # plot temporal cluster extent
    ymin, ymax = ax[1].get_ylim()
    ax[1].fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color="tab:grey", alpha=0.2
    )

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)

    return fig


def draw_sig_rectangle(t_start, t_stop, ax, alpha=0.2):
    from matplotlib.patches import Rectangle

    ylim = ax.get_ylim()
    width = t_stop - t_start
    height = ylim[1] - ylim[0]
    xy = (t_start, ylim[0])
    rectangle = Rectangle(xy, width, height, color="grey", alpha=alpha)
    ax.add_patch(rectangle)


# %%
# do the work

conds = ["harmonic", "inharmonic", "inharmonic_changing"]
oddballs = ["standard", "deviant_1", "deviant_2", "deviant_3"]

epochs = read_epochs(epochs_path, participants)
epochs.save(epochs_path / "all_epochs-epo.fif", overwrite=True)

# %%
evoked = extract_evokeds(epochs, participants)

# %%
# serialize this and pickle because other scripts use this data
with open("../results/evokeds/all.p", "wb") as f:
    pickle.dump(evoked, f)
# %%
evoked_arrays = evoked_ndarrays(evoked, conds)


# %%
# cluster-based permutations

# get adjacency matrix
adjacency_raw_path = str(raw_path / "0005/Inharmonic0005.vhdr")
adjacency, ch_names = get_adjacency_matrix(adjacency_raw_path)

conds = ["harmonic", "inharmonic", "inharmonic_changing"]
mismatches = [f"mismatch_{i}" for i in range(1, 4)]
contrasts = ["harm/ih", "ih/ic", "harm/ic"]

# Presence of MMN/P3
cluster_mismatches = calculate_cluster_mismatches(evoked_arrays, adjacency)


# F-test between three conditions
cluster_ftest = {}
for mismatch in mismatches:
    print(f"Running cluster-based f-test for {mismatch}...")
    X = [evoked_arrays[cond][mismatch] for cond in conds]
    cluster_ftest[mismatch] = calculate_clusters_ftest(X, adjacency, n_permutations=1e4)

# T-test for post-hoc contrasts
cluster_ttest = calculate_cluster_contrasts(evoked_arrays, adjacency)


# %%
# Presence of ORN
cluster_orn = calculate_cluster_orn(evoked_arrays, adjacency)

# %%
# pickle so that plotting scripts can make use of it
with open(clustering_path / "cluster_mismatches.p", "wb") as f:
    pickle.dump(cluster_mismatches, f)

with open(clustering_path / "cluster_ftest.p", "wb") as f:
    pickle.dump(cluster_ftest, f)

with open(clustering_path / "cluster_ttest.p", "wb") as f:
    pickle.dump(cluster_ttest, f)

with open(clustering_path / "cluster_orn.p", "wb") as f:
    pickle.dump(cluster_orn, f)
# %%
# print results
# %%
ftest_table_list = []
for m in mismatches:
    print(f"{m.capitalize()}:")
    print_sig_clusters(cluster_ftest[m])
    d = print_all_clusters(cluster_ftest[m])
    d['mismatch'] = m
    ftest_table_list.append(d)

ftest_table = pd.concat(ftest_table_list)
ftest_table.to_csv(project_path / "results" / 'ftest_clusters_table.csv')

# %%

for m in mismatches:
    print(f"\n* Mismatch {m}:")
    for c in contrasts:
        print(f"Contrast {c}:")
        print_sig_clusters(cluster_ttest[m][c], p_thresh=(0.05))

# %%
# %%
print("Harmonic vs inharmonic:")
print_sig_clusters(cluster_ttest["mismatch_1"]["harm/ih"], p_thresh=0.05 / 3)

print("Harmonic vs changing:")
print_sig_clusters(cluster_ttest["mismatch_1"]["harm/ic"], p_thresh=0.05 / 3)

print("Inharmonic vs changing")
print_sig_clusters(cluster_ttest["mismatch_1"]["ih/ic"], p_thresh=0.05 / 3)



# %%
# print significant clusters
conds = ["harmonic", "inharmonic", "inharmonic_changing"]
deviants = [f"deviant_{i}" for i in range(1, 4)]
for d in deviants:
    print(f"\n## {d}:\n")
    for c in conds:
        print(f"# Condition {c}:")
        print_sig_clusters(cluster_mismatches[d][c])

# %%
# print all clusters
all_clust_list = []
for d in deviants:
    for c in conds:
        all_clusters = print_all_clusters(cluster_mismatches[d][c])
        all_clusters['condition'] = c
        all_clusters['deviants'] = d
        all_clust_list.append(all_clusters)

all_clust = pd.concat(all_clust_list)
all_clust.to_csv(project_path / "results" / 'mismatch_clusters_table.csv')
all_clust

# %%
# print ORN clusters
print("ORN Analysis:")
all_clust_list = []
for i in ['standard', 'deviant_1']:
    for j in ['inharmonic', 'changing', 'contrast']:
        print(f"\n# {i.capitalize()}, {j}:")
        print_sig_clusters(cluster_orn[i][j])
        all_clusters = print_all_clusters(cluster_orn[i][j])
        all_clusters['condition'] = j
        all_clusters['deviance'] = i
        all_clust_list.append(all_clusters)

all_clust = pd.concat(all_clust_list)
all_clust.to_csv(project_path / "results" / 'orn_clusters_table.csv')
all_clust


# %%
# CLUSTER PLOTS
# mismatch 1
plot_ftest_clusters(evoked, cluster_ftest['mismatch_1'], mismatch="mismatch_1", cluster_id=0)
plot_ftest_clusters(evoked, cluster_ftest['mismatch_1'], mismatch="mismatch_1", cluster_id=1)
plot_ftest_clusters(evoked, cluster_ftest['mismatch_1'], mismatch="mismatch_1", cluster_id=2)

# %%
# mismatch 2
plot_ftest_clusters(evoked, cluster_ftest['mismatch_2'], mismatch="mismatch_2", cluster_id=0)

# %%
# mismatch 3
plot_ftest_clusters(evoked, cluster_ftest['mismatch_3'], mismatch="mismatch_3", cluster_id=0)

# %%
# Post-hoc pairwise comparisons
# Bonferoni corrected alpha level is .05 / 3 = .016666

cond_contrasts = {
    "harm/ih": ["harmonic", "inharmonic"],
    "ih/ic": ["inharmonic", "inharmonic_changing"],
    "harm/ic": ["harmonic", "inharmonic_changing"],
}
all_clust_list = []
for m in mismatches:
    for c in contrasts:
        for cluster_id in range(3):
            plot_ttest_clusters(
                evoked,
                cluster_ttest[m][c],
                conditions=cond_contrasts[c],
                mismatch=m,
                cluster_id=cluster_id,
                combine="mean",
            )
        all_clusters = print_all_clusters(cluster_ttest[m][c])
        all_clusters['condition'] = c
        all_clusters['mismatch'] = m
        all_clust_list.append(all_clusters)


all_clust = pd.concat(all_clust_list)
all_clust.to_csv(project_path / "results" / 'posthoc_clusters_table.csv')
all_clust
# %%
