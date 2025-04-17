# %%

import numpy as np
import pandas as pd
import antropy as ant
import soundfile as sf
import seaborn as sns
import matplotlib.pyplot as plt

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
    _, fs = sf.read("../metadata/sound_pool/h500.wav")
    return fs


# config
blocks = ["h", "i", "ic"]
block_ids = [1, 2]
pids = list(range(1, 36))
fs = get_fs()  # sample rate
pool_conds = ["h", "ih"]
pool_ids = range(0, 1000)
pool_freqs = np.arange(500, 850, 50)

# %%
dfs = []
for p in pids:
    print(f"Calculating entropy values for participant {p}...")
    for block in blocks:
        for bid in block_ids:
            print(f"For sequence {block}{bid}...")
            # open sequence file
            fname = f"../metadata/roving_sequences/p{p}_roves_{block}{bid}.csv"
            df = pd.read_csv(fname)
            sound_list = [load_sound(s) for s in df.filename]

            # calculate entropy for the whole sequence
            seq = np.concatenate(sound_list)
            sequence_ent = ant.app_entropy(seq)

            res = {
                "pid": p,
                "condition": block,
                "block_id": bid,
                "entropy": sequence_ent,
            }
            dfs.append(pd.DataFrame(res, index=[0]))

print("...saving to file...")
df = pd.concat(dfs, ignore_index=True)
df.to_csv("../results/entropies/sequence_entropies.csv")
# %%
# %%
# WARNING, this takes a long time

def calculate_entropy_measures():
# calculate entropies for single sounds
    pool_ids = range(0, 1000)
    # pool_ids = range(0, 100)

    res_list = []
    for f in pool_freqs:
        print(f'Calculating entropies for f = {f} Hz...')
        for id in pool_ids:
            s = load_sound(f"ih{f}_{id}.wav", fs)
            res = {
                "f": f,
                "id": id,
                "condition": "ih",
                "entropy": ant.app_entropy(s),
            }
            res_list.append(res)

    #  add harmonic
    for f in pool_freqs:
        s = load_sound(f"h{f}.wav", fs)
        res = {
            "f": f,
            "id": 0,
            "condition": "harm",
            "entropy": ant.app_entropy(s),
        }
        res_list.append(res)

    entropies_df = pd.DataFrame(res_list)
    return entropies_df

entropies_df = calculate_entropy_measures()
entropies_df.to_csv("../results/entropies/sound_entropies.csv")


# %%
entropies_df.groupby('condition').entropy.mean()
# %%
entropies_df.groupby('condition').entropy.std()
# %%
