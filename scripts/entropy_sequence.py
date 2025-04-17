# This script computes entropy values for all blocks in each experiment.
# Returns a pandas df containing entropy values for each block.

from sys import argv
import numpy as np
import pandas as pd
import antropy as ant
import soundfile as sf


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
fs = get_fs()  # sample rate
pool_conds = ["h", "ih"]
pool_ids = range(0, 1000)
pool_freqs = np.arange(500, 850, 50)


p = int(argv[1])
print(f"Calculating entropy values for participant {p}...")
dfs = []
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
output = pd.concat(dfs, ignore_index=True)
output.to_csv(f'../results/entropies/sequence_p{p}')
print('...done.')
