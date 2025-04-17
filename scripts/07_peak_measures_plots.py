# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from shared_functions import default_colors

conds = ["harmonic", "inharmonic", "changing"]


# def default_colors(conditions):
#     conds = ["harmonic", "inharmonic", "changing"]
#     colors = ["crimson", "tab:blue", "tab:olive"]
#     colors_dict = dict(zip(conds, colors))
#     ret = {c: colors_dict[c] for c in conditions}
#     return ret


# %%
# load peak measures
df = pd.read_csv("../results/peak_measures.csv")
df_orn = pd.read_csv("../results/peak_measures_orn.csv")
df_p2 = pd.read_csv("../results/peak_measures_p2.csv")


# recode "inharmonic changing" so it looks better in the plot
df.condition = df.condition.replace({"inharmonic_changing": "changing"})
df_p2.condition = df_p2.condition.replace({"inharmonic_changing": "changing"})
# %%
dff = []
dff.append(df.query("pitch_diff == 'all' and mismatch == 'mismatch_1'"))
dff.append(df.query("pitch_diff == 'all' and mismatch == 'mismatch_2'"))
dff.append(df.query("pitch_diff == 'all' and mismatch == 'mismatch_3'"))

# %%
measures = ["mmn_amp","mmn_peak_lat",  "p3_mean_amp", "p3_peak_lat"]


# %%
def plot_violins(df, measure, ax, contrasts, contrast_position):
    sns.lineplot(
        df,
        x="condition",
        y=measure,
        units="pid",
        estimator=None,
        alpha=0.2,
        color="black",
        linewidth=0.5,
        ax=ax,
    )
    sns.violinplot(df, x="condition", y=measure, palette=default_colors(conds), ax=ax)
    sns.scatterplot(
        df,
        x="condition",
        y=measure,
        # alpha=.5,
        marker=".",
        color="black",
        ax=ax,
    )

    # plot contrast lines
    line_y = contrast_position[0]
    line_y2 = contrast_position[0] + contrast_position[1] + 1
    xlim1 = (0.18, 0.47)
    xlim2 = (0.53, 0.80)
    xlim3 = (0.2, 0.82)

    if contrasts[0] is not None:
        ax.axhline(
            y=line_y,
            xmin=xlim1[0],
            xmax=xlim1[1],
            color="red",
            marker="|",
            markersize=10,
            markeredgewidth=2,
        )
        ax.text(
            x=0.5, y=line_y, s=contrasts[0], color="red", horizontalalignment="center"
        )

    if contrasts[1] is not None:
        ax.axhline(
            y=line_y2,
            xmin=xlim3[0],
            xmax=xlim3[1],
            color="red",
            marker="|",
            markersize=10,
            markeredgewidth=2,
        )
        ax.text(
            x=1, y=line_y2, s=contrasts[1], color="red", horizontalalignment="center"
        )

    if contrasts[2] is not None:
        ax.axhline(
            y=line_y,
            xmin=xlim2[0],
            xmax=xlim2[1],
            color="red",
            marker="|",
            markersize=10,
            markeredgewidth=2,
        )
        ax.text(
            x=1.5, y=line_y, s=contrasts[2], color="red", horizontalalignment="center"
        )


# %%


fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 12), sharex=True, sharey="row")

contrast_matrix = [
    [[None, "***", "***"], [None, None, None], [None, None, None]],
    [[None, None, None], [None, None, None], [None, None, None]],
    [["***", None, "***"], [None, "**", "**"], [None, None, None]],
    [[None, None, None], [None, None, None], [None, None, None]],
]

positioning_matrix = (
    ((6.3, 0.5), (2.2, 0.5), (2.4, 0.5)),
    ((0.37, 0.05), (0.33, 0.05), (0.34, 0.05)),
    ((8.5, 0.5), (6.5, 0.5), (4, 0.5)),
    ((0.17, 0.05), (0.33, 0.05), (0.34, 0.05)),
)

for i, m in enumerate(measures):
    for j, d in enumerate(dff):
        plot_violins(
            df=d,
            measure=m,
            ax=axs[i][j],
            contrasts=contrast_matrix[i][j],
            contrast_position=positioning_matrix[i][j],
        )


axs[0][0].set_ylabel("MMN Mean Amplitude (\u03bcV)")
axs[1][0].set_ylabel("MMN Peak Latency (s)")
axs[2][0].set_ylabel("P3 Mean Amplitude (\u03bcV)")
axs[3][0].set_ylabel("P3 Peak Latency (s)")

for i in range(3):
    axs[3][i].set_xlabel(None)

axs[0][0].set_title("1st deviant")
axs[0][1].set_title("2nd deviant")
axs[0][2].set_title("3rd deviant")

fig.supxlabel("Condition")
fig.tight_layout()
fig.show()
plt.savefig("../results/plots/fig_s1.png", dpi=300)

# %%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharex=True, sharey=False)

contrast_matrix = [
    [[None, "***", "***"], ["***", None, "***"]],
    [[None, None, None], [None, None, None]],
]

positioning_matrix = (
    ((3.3, 0.5), (8.5, 0.5)),
    ((8.5, 0.5), (6.5, 0.5)),

)

# mmn mean amp
i, j = 0, 0
plot_violins(
    df=dff[0],
    measure=measures[0],
    ax=axs[i][j],
    contrasts=contrast_matrix[i][j],
    contrast_position=positioning_matrix[i][j],
)

# mmn peak latency
i, j = 1, 0
plot_violins(
    df=dff[0],
    measure=measures[1],
    ax=axs[i][j],
    contrasts=contrast_matrix[i][j],
    contrast_position=positioning_matrix[i][j],
)

# p3 amplitude
i, j = 0, 1
plot_violins(
    df=dff[0],
    measure=measures[2],
    ax=axs[i][j],
    contrasts=contrast_matrix[i][j],
    contrast_position=positioning_matrix[i][j],
)


# p3 latency
i, j = 1, 1
plot_violins(
    df=dff[0],
    measure=measures[3],
    ax=axs[i][j],
    contrasts=contrast_matrix[i][j],
    contrast_position=positioning_matrix[i][j],
)


axs[0][0].set_ylabel("MMN Mean Amplitude (\u03bcV)")
axs[1][0].set_ylabel("MMN Peak Latency (s)")
axs[0][1].set_ylabel("P3a Mean Amplitude (\u03bcV)")
axs[1][1].set_ylabel("P3a Peak Latency (s)")

axs[1][1].set_xlabel(None)
axs[1][0].set_xlabel(None)

fig.supxlabel("Condition")
fig.tight_layout()
fig.show()
plt.savefig("../results/plots/fig_4.png", dpi=300)
# %%
# ORN
sns.violinplot(data=df_orn, x="ds", y="orn_mean_amp", hue="condition")

# %%
sns.violinplot(data=df_orn, x="ds", y="orn_peak_lat", hue="condition")



# %%
#P2
sns.boxplot(data=df_p2, x='condition', y='p2_amp', hue='deviance')


# %%
sns.violinplot(data=df_p2, x='condition', y='p2_amp', hue='deviance')



# %%
# pitch deviants
sel = (df.pitch_diff != 'all') & (df.mismatch == 'mismatch_1')
dfpd = df[sel]

fig, axs = plt.subplots(2, 2,figsize=(10,8), sharex=True,)

sns.boxplot(data=dfpd, x='pitch_diff', y='mmn_mean_amp', hue='condition', palette=default_colors(conds), ax=axs[0][0])

sns.boxplot(data=dfpd, x='pitch_diff', y='p3_mean_amp', hue='condition', palette=default_colors(conds), ax=axs[0][1])

sns.boxplot(data=dfpd, x='pitch_diff', y='mmn_peak_lat', hue='condition', palette=default_colors(conds), ax=axs[1][0])

sns.boxplot(data=dfpd, x='pitch_diff', y='p3_peak_lat', hue='condition', palette=default_colors(conds), ax=axs[1][1])

axs[0][0].set_ylabel("MMN Mean Amplitude (\u03bcV)")
axs[1][0].set_ylabel("MMN Peak Latency (s)")
axs[0][1].set_ylabel("P3a Mean Amplitude (\u03bcV)")
axs[1][1].set_ylabel("P3a Peak Latency (s)")

axs[0][1].set_xlabel(None)
axs[0][0].set_xlabel(None)
axs[1][1].set_xlabel("Frequency shift")
axs[1][0].set_xlabel("Frequency shift")

axs[0][1].get_legend().remove()
axs[1][1].get_legend().remove()
axs[1][0].get_legend().remove()

fig.tight_layout()
fig.show()
plt.savefig("../results/plots/fig_s2.png", dpi=300)
# %%
