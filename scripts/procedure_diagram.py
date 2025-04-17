# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches





def harmonics(f0, n=None):
    '''Return a harmonic set starting from f0.
    n - number of harmonics, if none continue to 22kHz'''
    if n:
        return [i * f0 for i in np.arange(1, n+1)]
    else:
        return [i for i in np.arange(f0, 24001, f0)]


def make_jitter_rates(f0, rate=.5):
    """Provide a list of jitter rates for a given f0."""


    # make a harmonic series
    freqs = harmonics(f0)

    # set iterators
    i = 0
    jitter_rates = []

    # loop until all frequencies have viable jitter rates
    # rejection sampling until |f(n) - f(n-1)| > 30
    while i < len(freqs):
        if i == 0:
            # for f0 its always 1
            jitter_rates.append(0)
            i += 1
        else:
            # jitter everything else
            jitter_rate = (np.random.random(1)[0] * 2 * rate) - rate
            jitter_f = freqs[i] + (jitter_rate * freqs[0])
            # rejection sampling
            if (abs(jitter_f - freqs[i-1]) >= 30):
                jitter_rates.append(jitter_rate)
                i += 1

    return jitter_rates


def apply_jitter(input, rate=.5):
    jitter_rates = make_jitter_rates(input[0], rate)
    jit = jitter_rates[:len(input)]
    changes = jit * input
    res = input + changes
    return res


# %%
st_f0 = 500
dev_f0 = 650

harm = 8

standard = np.arange(st_f0, (harm+1)*st_f0, st_f0)
deviant = np.arange(dev_f0, (harm+1)*dev_f0, dev_f0)

deviant


# %%
fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(15,4))

# draw harmonic stimuli
y_s = np.concatenate((np.tile(standard, (3,1)), np.tile(deviant, (2, 1))))
x_s = np.reshape(np.repeat([2,3,4,6,7],harm), (harm,5))
axs[0].scatter(x_s, y_s, s=200, c='b', marker='_')

x_d = np.reshape(np.repeat([1,5], harm), (harm,2))
y_d = np.reshape(np.concatenate((standard, deviant)), (harm,2))
axs[0].scatter(x_d, y_d, s=200, c='r', marker='_')


# draw inharmonic stimuli
jitter_rate = .2
ih_standard = apply_jitter(standard, jitter_rate)
ih_deviant = ih_standard + 150
y_s = np.concatenate((np.tile(ih_standard, (3,1)), np.tile(ih_deviant, (2, 1))))
x_s = np.reshape(np.repeat([2,3,4,6,7],harm), (harm,5))
axs[1].scatter(x_s, y_s, s=200, c='b', marker='_')

x_d = np.reshape(np.repeat([1,5], harm), (harm,2))
y_d = np.reshape(np.concatenate((ih_standard, ih_deviant)), (harm,2))
axs[1].scatter(x_d, y_d, s=200, c='r', marker='_')


# draw inharmonic changing stimuli
y_list, x_list = [], []
for i in range(2, 5):
    y_list.append(apply_jitter(standard, jitter_rate))
    x_list.append(np.repeat([i], harm))

for i in range(6, 8):
    y_list.append(apply_jitter(deviant, jitter_rate))
    x_list.append(np.repeat([i], harm))

y_s = np.concatenate(y_list)
x_s = np.concatenate(x_list)

axs[2].scatter(x_s, y_s, s=200, c='b', marker='_')

x_d = np.reshape(np.repeat([1, 5], harm), (harm, 2))
ih_standard = apply_jitter(standard, jitter_rate)
ih_deviant = apply_jitter(deviant, jitter_rate)
y_d = np.reshape(np.concatenate((ih_standard, ih_deviant)), (harm,2))
axs[2].scatter(x_d, y_d, s=200, c='r', marker='_')


# Create a Rectangle patch
rect = patches.Rectangle((.5, 400), 7, 400, linewidth=1,
                         edgecolor='grey', facecolor='none')
# Add the patch to the Axes
axs[0].add_patch(rect)


# Create a Rectangle patch
rect = patches.Rectangle((.5, 400), 7, 400, linewidth=1,
                         edgecolor='grey', facecolor='none')
axs[1].add_patch(rect)


# Create a Rectangle patch
rect = patches.Rectangle((.5, 400), 7, 400, linewidth=1,
                         edgecolor='grey', facecolor='none')
axs[2].add_patch(rect)

top_limit = 4e3
axs[0].set_ylim(top=top_limit)
axs[1].set_ylim(top=top_limit)
axs[2].set_ylim(top=top_limit)

# make nice-looking axes
axs[0].set_xlim((0, 8))
axs[0].set_xticks(list(range(1,8)))
axs[1].set_xlim((0, 8))
axs[1].set_xticks(list(range(1,8)))
axs[2].set_xlim((0, 8))
axs[2].set_xticks(list(range(1,8)))



axs[0].set_title('Harmonic')
axs[1].set_title('Inharmonic')
axs[2].set_title('Inharmonic changing')


axs[0].set_ylabel('frequency (Hz)')
axs[1].set_xlabel('stimulus no.')


plt.savefig('procedure_diagram.pdf', dpi=150)


# %%
