import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import mne
import numpy as np


# channel picks for plotting
channel_picks = ["F3", "Fz", "F4", "FC1", "FC2"]

# default plot folder
plot_fpath = "../results/plots/"


def read_raw(p, raw_path, project_path):
    # set aliases for electrode positions
    # for participants 1 and 2 (mistake in channel naming)
    if p == 2:
        montage_aliases = {"O2": "Oz", "O3": "O2", "MastL": "M1", "MastR": "M2"}
    elif p == 1:
        montage_aliases = {
            "O2": "Oz",
            "O3": "O2",
        }
    else:
        montage_aliases = {
            # 'O2': 'Oz',
            # 'O3': 'O2',
            "MastL": "M1",
            "MastR": "M2",
        }

    # load fpaths
    # participant info table
    participant_info = pd.read_csv(
        project_path / "metadata" / "participant_info.csv", index_col="pid"
    )
    filepaths = participant_info["eeg_path_local"]

    # load data
    eeg_path = str(raw_path) + participant_info["eeg_path_local"][p]
    raw = mne.io.read_raw_brainvision(
        raw_path / eeg_path, eog=["CanEye", "LowEye"], misc=["Heart"], preload=True
    )

    # downsample for participant 1
    if p == 1:
        raw = raw.resample(1000)

    if p < 3:
        mne.rename_channels(raw.info, montage_aliases)
    # set montage
    raw.set_montage("standard_1020")

    return raw


def apply_filters(raw, low, high, notch):
    # band-pass
    raw.filter(l_freq=low, h_freq=high)

    # notch filter for line noise
    notch_freqs = np.arange(notch, raw.info["sfreq"] / 2, notch)
    raw.notch_filter(notch_freqs)


def config():
    c = {
        # project file path
        "project_path": pathlib.Path("/Users/kbas/cloud/sci/harmonicity_roving"),
        # raw data filepath
        "raw_path": pathlib.Path("/Users/kbas/sci_data/harmonicity_data/raw"),
        # epochs filepath`
        "epochs_path": pathlib.Path("/Users/kbas/sci_data/harmonicity_data/epochs"),
    }

    # ICA results path
    c["ica_path"] = c["project_path"] / "results" / "ica"

    # autoreject objects path
    c["ar_path"] = c["project_path"] / "results" / "autoreject"

    return c


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


def default_colors(conditions):
    colors_dict = {
        "harmonic": "crimson",
        "harmonic standard": "crimson",
        "harmonic deviant": "crimson",
        "inharmonic": "tab:blue",
        "inharmonic standard": "tab:blue",
        "inharmonic deviant": "tab:blue",
        "ORN inharmonic": "tab:blue",
        "inharmonic_changing": "tab:olive",
        "changing": "tab:olive",
        "changing standard": "tab:olive",
        "changing deviant": "tab:olive",
        "ORN changing": "tab:olive",
        "ORN": "green",
        "standard": "blue",
        "deviant": "red",
        "deviant#1": "red",
        "deviant#2": "red",
        "deviant#3": "red",
        "mismatch": "green",
        "mmn": "green",
    }

    # create new colors_dict with uppercase keys
    upper_dict = {k.capitalize(): v for k, v in colors_dict.items()}
    colors = colors_dict | upper_dict
    ret = {c: colors[c] for c in conditions}
    return ret


def plot_evoked_comparison(
    comparison,
    ax=None,
    show_sensors=True,
    show_legend=True,
    picks=channel_picks,
    title=None,
    save_file=None,
    ylim=None,
    clusters=None
):
    colors = default_colors(comparison.keys())
    if title is None:
        title = ""
    mne.viz.plot_compare_evokeds(
        comparison,
        picks,
        colors=colors,
        show_sensors=show_sensors,
        axes=ax,
        title=title,
        show=False,
        combine="mean",
        sphere="eeglab",
        legend=show_legend,
        ylim=ylim,
    )

    # plot temporal cluster extent
    if clusters is not None:
        for c in clusters:
            ymin, ymax = ax.get_ylim()
            ax.fill_betweenx(
                (ymin, ymax), c[0], c[1], color="tab:grey", alpha=0.2
            )

    if save_file is not None:
        plt.savefig(f"{plot_fpath}{save_file}", dpi=300)


def plot_ttest_clusters2(
    evokeds,
    clustering_results,
    title=None,
    cluster_id=0,
    p_cutoff=0.05,
    combine="gfp",
    ch_inds=None,
):
    """Plots clustering results from cluster-based permutations (t-test comparing two conditions).

    Args:
        evokeds (dict): A dict of lists of evoked responses. Dict keys determine plot labels.
        clustering_results (tuple): Results of cluster-based permutations. A tuple contains t_obs, clusters, p_values, h0. This can taken from output of mne.stats.spatio_temporal_cluster_1samp_test.
        cluster_id (int, optional): Id of the cluster to plot. This is an id from a list of consecutive, statistically significant clusters. Defaults to 0.
        p_cutoff (float, optional): P-value to cutoff significant clusters. Defaults to 0.
        combine (string): Channel combine method for traces, can be 'gfp' or 'mean'. Defaults to gfp.
        ch_inds (Array-like): Which channels to plot. If None, plot the "significant" cluster channels. Defaults to None.

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

    colors = default_colors(conditions)

    # get cluster times
    sig_times = evokeds[conditions[0]][1].times[t_mask]

    # pick significant cluster channels
    if ch_inds is None:
        ch_inds = np.squeeze(mask).nonzero()[0]
        show_sensors = None
    else:
        show_sensors = True

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
    ax[0].set_xlabel(
        f"Averaged t-map \n ({sig_times[0]:.3f} - {sig_times[-1]:.3f} s) \np = {pv:.3f}"
    )

    # plot evoked traces
    mne.viz.plot_compare_evokeds(
        evokeds,
        title=title,
        picks=ch_inds,
        axes=ax[1],
        colors=colors,
        show=False,
        split_legend=True,
        truncate_yaxis="auto",
        combine=combine,
        show_sensors=show_sensors,
    )

    # plot temporal cluster extent
    ymin, ymax = ax[1].get_ylim()
    ax[1].fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color="tab:grey", alpha=0.2
    )

    # clean up viz
    # mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)

    return fig


def plot_clusters(
    evokeds,
    clustering_results,
    p_cutoff=0.05,
    title=None,
    combine="mean",
    ch_inds=None,
    ylims=None,
    topo_vlim=None,
    map_colors=None
):
    from matplotlib import ticker

    # unpack from clustering results
    t_obs, clusters, p_values, _ = clustering_results

    # pick one cluster to plot
    good_clusters_idx = np.where(p_values < p_cutoff)[0]
    good_clusters = [clusters[i] for i in good_clusters_idx]

    # get the cluster p-value for plotting
    pv = [p_values[i] for i in good_clusters_idx]

    # get the info object from the first dict element
    conditions = list(evokeds.keys())
    evokeds_info = evokeds[conditions[0]][0].info

    width_ratios = [10] * 3 + [50]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=4,
        gridspec_kw={"width_ratios": width_ratios},
        figsize=(16, 4),
    )

    if map_colors is None:
        map_colors_use = 'RdBu_r'
    else:
        map_colors_use = map_colors
    
    if topo_vlim is None:
        topo_vlim_use = (np.min, np.max)
    else:
        topo_vlim_use = (topo_vlim[0]*100000, topo_vlim[1]*100000)

    for j, cluster in enumerate(good_clusters):
        i = 3 - len(good_clusters) + j
        # get topography for F/t stat
        # average over the timecourse of the cluster
        t_map = (t_obs * cluster).mean(axis=0)

        # create spatial mask
        mask = (cluster.sum(axis=0) > 0)[:, np.newaxis]

        # create time mask
        t_mask = cluster.sum(axis=1) > 0

        # get cluster times
        sig_times = evokeds[conditions[0]][1].times[t_mask]

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(t_map[:, np.newaxis], evokeds_info, tmin=0)


        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            cmap=map_colors_use,
            vlim=topo_vlim_use,
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
            axes=ax[i],
        )

        # remove the title that would otherwise say "0.000 s"
        ax[i].set_title("")

        ax[i].set_xlabel(
            f"{sig_times[0]:.3g} - {sig_times[-1]:.3g} s \np = {pv[j]:.4g}"
        )

        # add axes for colorbar
        plt.colorbar(
            ax[i].images[0],
            ax=ax[i],
            format=ticker.FuncFormatter(lambda x, pos: round(x / 100000)),
            shrink=0.4,
        )

    # turn off axes that are unnecessary
    for i in range(3):
        if i < 3 - len(good_clusters):
            ax[i].axis('off')

    combine = "mean"
    show_sensors = True
    if ylims is not None:
        ylims_d = {"eeg": ylims}
    else:
        ylims_d = ylims

    # plot evoked traces
    mne.viz.plot_compare_evokeds(
        evokeds,
        title="",
        picks=channel_picks,
        axes=ax[-1],
        colors=default_colors(evokeds.keys()),
        show=False,
        split_legend=True,
        truncate_yaxis="auto",
        combine=combine,
        show_sensors=show_sensors,
        ylim=ylims_d,
    )

    # plot temporal cluster extent
    ymin, ymax = ax[-1].get_ylim()

    for i, cluster in enumerate(good_clusters):
        # create spatial mask
        mask = (cluster.sum(axis=0) > 0)[:, np.newaxis]

        # create time mask
        t_mask = cluster.sum(axis=1) > 0

        # get cluster times
        sig_times = evokeds[conditions[0]][1].times[t_mask]

        ax[-1].fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="tab:grey", alpha=0.2
        )

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    # fig.subplots_adjust(bottom=0.05)

    fig.suptitle(title)
    return fig


def draw_sig_rectangle(t_start, t_stop, ax, alpha=0.2):
    from matplotlib.patches import Rectangle

    ylim = ax.get_ylim()
    width = t_stop - t_start
    height = ylim[1] - ylim[0]
    xy = (t_start, ylim[0])
    rectangle = Rectangle(xy, width, height, color="grey", alpha=alpha)
    ax.add_patch(rectangle)


def save_plot(fname):
    plt.savefig(f"../results/plots/{fname}", dpi=300, bbox_inches="tight")


def stack_images(img_fpaths, output_file, orientation="vertical", padding="left"):
    """Takes a bunch of image files, stacks them horizontally or vertically and saves them as a new file. Uses PIL/pillow under the hood for image processing.

    Args:
        img_fpaths (list): List of filenames containing images to stack.
        output_file (str): Filename of the output.
        orientation (str, optional): Stacking orientation. Can be 'horizontal' or 'vertical'. Defaults to 'vertical'.
        padding (str, optional): If stacking vertically, should the images be padded to the 'left' or 'right'. Defaults to 'left'.
    """
    # import PIl
    from PIL import Image

    # load images in
    imgs = [Image.open(i) for i in img_fpaths]

    # calculate size of new canvas
    size_array = np.array([i.size for i in imgs])
    if orientation == "vertical":
        x_dim = size_array[:, 0].max()
        y_dim = size_array[:, 1].sum()
    elif orientation == "horizontal":
        x_dim = size_array[:, 0].sum()
        y_dim = size_array[:, 1].max()
    new_size = (x_dim, y_dim)

    # make new canvas
    out = Image.new("RGB", new_size, color="white")

    # initialize my reference point
    x, y = 0, 0
    # paste images to new canvas
    for i in imgs:
        if padding == "right":
            x = x_dim - i.size[0]
        out.paste(i, (x, y))
        # figure out the placement of next image
        if orientation == "vertical":
            y += i.size[1]
        elif orientation == "horizontal":
            x += i.size[0]

    # write to file
    out.save(output_file)
    print(f"Output saved successfully under {output_file}")


def identify_significant_clusters(clusters, time_vector, return_spaces=False, p_cutoff = .05):
    # unpack from clustering results
    _, clusters, p_values, _ = clusters

    # pick one cluster to plot
    good_clusters_idx = np.where(p_values < p_cutoff)[0]
    good_clusters = [clusters[i] for i in good_clusters_idx]

    times, spaces = [], []

    for i, cluster in enumerate(good_clusters):
        # create spatial mask
        mask = (cluster.sum(axis=0) > 0)[:, np.newaxis]
        spaces.append(mask)

        # create time mask
        t_mask = cluster.sum(axis=1) > 0

        # get cluster times
        sig_times = time_vector[t_mask]
        cluster_times = (sig_times[0], sig_times[-1])
        times.append(cluster_times)

    if return_spaces:
        return times, spaces
    else:
        return times    


