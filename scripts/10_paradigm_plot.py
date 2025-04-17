# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf

# %%

def load_sound(fname, pad=True, ioi=0.5):
    # open sound
    sound, fs = sf.read(f"../paradigms/eeg/sound_pool/{fname}")

    if pad:
        # pad the sound so the ioi is 500 ms.
        pad_width = int((fs * ioi) - sound.shape[0])
        sound_padded = np.pad(sound, (0, pad_width))
        return sound_padded
    else:
        return sound
    # return padded sound


def get_fs():
    """Look at a sample file to get the sample rate."""
    _, fs = sf.read("../paradigms/eeg/sound_pool/h500.wav")
    return fs


# paradigm config
# train of frequencies (f0's)
train = [600] * 5 + [700] * 7 + [500] * 8 + [650] * 5
deviants_1 = np.array([0, 5, 12, 20])
deviants_2 = deviants_1 + 1
deviants_3 = deviants_1 + 2

no_partials = 12
jitter_strength = .3

rng = np.random.default_rng()
t = train
m = np.arange(1, no_partials + 1, 1)

# tile
t_matrix = np.tile(t, (no_partials, 1))
# aim: get ndarray shape (4,8)
m_matrix = np.tile(m, (len(t), 1)).T

stims = m_matrix * t_matrix

# make full jitter matrix
jitter_matrix = rng.uniform(-jitter_strength, jitter_strength, stims.shape)

# leave the f0 alone
jitter_matrix[0,:] = 0

# apply full jitter matrix to changing condition
y_changing = stims + (stims*jitter_matrix)

# apply first column of the jitter matrix to inharmonic condition
jitter_inharm = np.tile(jitter_matrix[:, 0], (len(t), 1)).T
y_inharm = stims + (stims * jitter_inharm)

y_s = stims

x_s = np.tile(np.arange(1, len(train)+1), (no_partials, 1))

# entropies
entropies_df = pd.read_csv("../results/entropies/sound_entropies.csv")

# spectra
fs = get_fs()
sig_h = load_sound(f"h500.wav", fs)
sig_ih = load_sound(f"ih500_21.wav", fs)

freq_ticks = [500, 1000, 2000, 5000, 10000, 20000]
freq_labels = [500, '1k', '2k', '5k', '10k', '20k']
time_ticks = [.005, .01, .015, .02]
time_labels = [5, 10, 15, 20]
time_vector = np.arange(len(sig_h)) / fs

# %%
# START PLOTTING
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(14,8))

# plot standards
axs[0][0].scatter(x_s, y_s, marker='_', c='grey')
axs[0][1].scatter(x_s, y_inharm, marker='_', c='grey')
axs[0][2].scatter(x_s, y_changing, marker='_', c='grey')


# highlight fundamental freqs
axs[0][0].scatter(x_s[0,:], y_s[0,:], marker='_', c='blue')
axs[0][1].scatter(x_s[0,:], y_inharm[0,:], marker='_', c='blue')
axs[0][2].scatter(x_s[0,:], y_changing[0,:], marker='_', c='blue')

# highlight deviants
# first order
axs[0][0].scatter(x_s[:, deviants_1], y_s[:, deviants_1], marker='_', c='red', alpha=1)
axs[0][1].scatter(x_s[:, deviants_1], y_inharm[:, deviants_1], marker='_', c='red', alpha=1)
axs[0][2].scatter(x_s[:, deviants_1], y_changing[:, deviants_1], marker='_', c='red', alpha=1)

# second-order
axs[0][0].scatter(x_s[:, deviants_2], y_s[:, deviants_2], marker='_', c='red', alpha=.6)
axs[0][1].scatter(x_s[:, deviants_2], y_inharm[:, deviants_2], marker='_', c='red', alpha=.6)
axs[0][2].scatter(x_s[:, deviants_2], y_changing[:, deviants_2], marker='_', c='red', alpha=.6)

# third-order
axs[0][0].scatter(x_s[:, deviants_3], y_s[:, deviants_3], marker='_', c='red', alpha=.3)
axs[0][1].scatter(x_s[:, deviants_3], y_inharm[:, deviants_3], marker='_', c='red', alpha=.3)
axs[0][2].scatter(x_s[:, deviants_3], y_changing[:, deviants_3], marker='_', c='red', alpha=.3)

xtick = [1, 5, 10, 15, 20, 25]
ytick = list(range(1000, 6000, 1000))
ytick_labels = [f'{i}k' for i in range(1, 6)]

# apply to all paradigm plots
for i in range(3):
    axs[0][i].set_ylim(300, 5000)
    axs[0][i].set_xlabel('Stimulus #')
    axs[0][i].set_ylabel('Frequency (Hz)')
    axs[0][i].set_xticks(xtick)
    axs[0][i].set_yticks(ytick, ytick_labels)




# entropy plot
sns.violinplot(entropies_df, y='entropy', x='f', hue='condition', legend=False, ax=axs[0][3])
axs[0][3].set_xlabel('Frequency (Hz)')
axs[0][3].set_ylabel('Entropy')
axs[0][3].get_legend().set_visible(False)

axs[1][0].plot(time_vector, sig_h)
axs[1][0].set_ylim(-.3, .3)
axs[1][0].set_ylabel('Amplitude (a.u.)')

Pxx, freqs, bins, im = axs[1][1].specgram(sig_h, NFFT=2**12,Fs=fs)
axs[1][1].set_yscale('log')
axs[1][1].set_ylim(400, 20000)
axs[1][1].set_yticks(freq_ticks, freq_labels)
axs[1][1].set_ylabel('Frequency (Hz)')

axs[1][2].plot(time_vector, sig_ih)
axs[1][2].set_ylim(-.3, .3)
axs[1][2].set_ylabel('Amplitude (a.u.)')

Pxx, freqs, bins, im = axs[1][3].specgram(sig_ih, NFFT=2**12,Fs=fs)
axs[1][3].set_yscale('log')
axs[1][3].set_ylim(400, 20000)
axs[1][3].set_yticks(freq_ticks, freq_labels)
axs[1][3].set_ylabel('Frequency (Hz)')

# apply to all lower-tier subplots
for i in range(4):
    axs[1][i].set_xticks(time_ticks, time_labels)
    axs[1][i].set_xlim(0.005, .02)
    axs[1][i].set_xlabel("Time (ms)")

# set subplot titles
subplot_titles = np.array([
    ['A) Harmonic', 'B) Inharmonic', 'C) Changing', 'D) Entropy'],
    ['E) Harmonic sound waveform', 'F) Harmonic sound spectrum', 'G) Inharmonic sound waveform', 'H) Inharmonic sound spectrum']
])
for i in range(2):
    for j in range(4):
        axs[i][j].set_title(subplot_titles[i][j], loc='left')

plt.tight_layout(w_pad=0)
plt.savefig('../results/plots/fig1.png', dpi=300)
fig.show()


# %%
