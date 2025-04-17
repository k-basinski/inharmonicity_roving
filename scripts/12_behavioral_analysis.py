# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shared_functions as sf

data_fpath = '../paradigms/behavioral/data/'
figpath = "../results/plots/"
plist = list(range(1,16))


# %%
# load data
dfs = []
for pid in plist:
    df = pd.read_csv(f'{data_fpath}{pid}_demography.csv')
    dfs.append(df)

demo = pd.concat(dfs)

dfs = []
for pid in plist:
    df = pd.read_csv(f'{data_fpath}responses_{pid}.csv')
    dfs.append(df)

data = pd.concat(dfs)

# do some cleaning
data.response = data.response.replace({
    'nie wiem': pd.NA
})
data.response = pd.to_numeric(data.response)

data.condition = data.condition.replace({
    'h': 'harmonic',
    'i': 'inharmonic',
    'ic': 'changing'
})

dt1 = data[data.task == 't1'].copy()
dt2 = data[data.task == 't2'].copy()

dt1.response = dt1.response.replace({
    1: 'One sound',
    2: 'Two sounds',
    3: "Three or more sounds"
})

# calculate difference between response and ground truth
dt2['error'] = dt2['response'] - dt2['gt']
dt2['abs_error'] = dt2.error.abs()

# drop to csv
dt1.to_csv(f'{data_fpath}behavioral_task1.csv')
dt2.to_csv(f'{data_fpath}behavioral_task2.csv')
# %%
# summarize demography
print(f'Median age was {demo.Wiek.median().round(2)}, SD = {demo.Wiek.std().round(2)}.')
print('Gender percentages:')
print(demo.Plec.value_counts())
# %%

# %%

# look at errors between conditions
# %%
condition_list = ['harmonic', 'inharmonic', 'changing']
colors = sf.default_colors(condition_list)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))

# Task 1 results
sns.countplot(dt1, x='condition', hue='response', ax=axs[0], order=condition_list)

# Task 2 results
sns.boxplot(data=dt2, x='condition', y='abs_error', palette=colors, order=condition_list, ax=axs[1])

axs[0].set_ylabel('no. responses')
axs[0].set_title('A) Task 1 results')
axs[1].set_ylabel('|error|')

plt.savefig(f'{figpath}behavioral_tasks.png', dpi=300)

# %%
