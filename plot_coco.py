import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


colors = {
    'baseline': '#9c9c9c',
    'BPTT1': '#d68383',
    'BPTT2': '#d44646',
    'BPTT3': '#d44646',
    'BPTT5': '#ad0000',
    'CBP5': '#9ac3fc',
    'CBP10': '#1d6fe0',
    'CBP20': '#004099',
    'RBP20': '#a632a8',
    'RBP20tanh': '#aa5eab',
    'RBP20sigmoid': '#d598d6',
}
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Plot best PQ results
df_50 = pd.read_csv('panoptic_50layer.csv')
df_101 = pd.read_csv('panoptic_101layer.csv')
df_50.resnet = 50
df_101.resnet = 101
df = pd.concat((df_50, df_101))
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(
    data=df,
    x='memory (MB)',
    y='PQ',
    hue='model',
    s=100,
    palette=colors,
    legend=False,
    ax=axs,
    marker='h')
plt.show()
plt.close(fig)

# df_full = df[df.model != 'RBP20']
# df_full['memory (MB)'] = df_full['memory (MB)'] / 1024
# df_rbp = df[df.model == 'RBP20']
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(
    data=df_50,
    x='memory (MB)',
    y='PQ',
    hue='model',
    s=100,
    palette=colors,
    legend=False,
    ax=axs[0],
    marker='h')
axs[0].set_title('Monotonic improvements in PQ with CBP timesteps')
axs[0].set_xlim([8500, 17500])
sns.scatterplot(
    data=df_101,
    x='memory (MB)',
    y='PQ',
    hue='model',
    s=100,
    palette=colors,
    legend=False,
    ax=axs[1],
    marker='h')
axs[1].set_title('Monotonic improvements in PQ with CBP timesteps')
axs[1].set_ylim([43, 44])
plt.show()
plt.close(fig)

# Plot PQ timecourses
ts_df = pd.read_csv('ts_data.csv')
fig, axs = plt.subplots(1, 1, figsize=(5, 3))
sns.lineplot(
    data=ts_df,
    x='timesteps',
    y='PQ',
    hue='model',
    marker='h',
    lw=1,
    palette=colors,
    legend=False)
plt.show()
plt.close(fig)
