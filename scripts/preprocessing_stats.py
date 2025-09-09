# %%
import matplotlib.pyplot as plt
import numpy as np

import shared_functions as sfuns
import autoreject
import pandas as pd
# get configuration
config = sfuns.config()

# %%

bad_epochs = []
all_epochs = []
logs = []
participants = list(range(1, 27)) + list(range(28, 38))


def percentage(x, y):
    perc = (x / y) * 100
    return round(perc, 1)


def get_droplogs():
    for p in participants:
        # open autoreject droplogs
        log = autoreject.read_reject_log(config['ar_path'] / f"{p}-rl2.npz")
        logs.append(log)

        # get bad and all epoch counts
        bad_count = log.bad_epochs.sum()
        all_count = log.bad_epochs.shape[0]
        bad_epochs.append(bad_count)
        all_epochs.append(all_count)


def plot_droplogs():
    for p, log in zip(participants, logs):
        print(f'Plotting droplogs for participant {p}...')

        # plot the droplog
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        ax.set_title(f'Participant {p}')
        log.plot(orientation='horizontal', aspect='auto', ax=ax, show=False)
        plt.savefig(config['ar_path'] / 'droplog_plots' / f"{p}-droplog.png", dpi=300)
        plt.close()


def bad_epoch_percentages():
    for p, b, a in zip(range(1, 38), bad_epochs, all_epochs):
        perc = percentage(b, a)
        if perc > 25:
            achtung = ' !!! '
        else:
            achtung = ''
        print(f'For participant {p}:')
        print(f'Bad epochs: {b} / {a} ({perc}%) {achtung}\n')
    # get overall percentage
    all_epochs_sum = np.array(all_epochs).sum()
    all_bads_sum = np.array(bad_epochs).sum()
    print(f'For the entire study: {all_bads_sum} / {all_epochs_sum} ({percentage(all_bads_sum, all_epochs_sum)}%)')


get_droplogs()
plot_droplogs()
bad_epoch_percentages()

# %%

res_list = []
blocks = ['h', 'i', 'ic']
p, block, bid = 1, 'h', 1
for p in range(1,36):
    for block in blocks:
        for bid in [1,2]:
            if block == 'h':
                s = 1
            else:
                s = 2
            fname = f"../metadata/roving_sequences/p{p}_roves_{block}{bid}.csv"
            df = pd.read_csv(fname)
            df['f'] = df.filename.apply(lambda x: x[s:s+3])
            df['f'] = pd.to_numeric(df['f'])
            # number of deviants
            deviants = np.sum(np.diff(df.f) != 0)
            res = {'p': p, 'block': block, 'bid': bid, 'dev': deviants}
            res_list.append(res)
# %%
df = pd.DataFrame(res_list)
df.dev.mean()
df.dev.std()
# %%
