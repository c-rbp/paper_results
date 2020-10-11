import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
# from matplotlib import colors


def colored_pf(data, start_ts, end_ts):
    """Make a colored timecourse for prediction."""
    canvas = np.zeros_like(data['readout'][0])
    for idx in range(start_ts, end_ts):
        canvas += ((data['readout'][idx] > 0).astype(canvas.dtype)) * (idx + 1)
    return canvas


# sns.set_style("white", {"axes.facecolor": ".9"})
bptt_no_pen = np.load('BPTT_No_Penalty.p', allow_pickle=True)
bptt_pen = np.load('BPTT_Penalty.p', allow_pickle=True)
cbp = np.load('CBP.p', allow_pickle=True)

scores_bptt_no_pen_all = bptt_no_pen['all_transformed_pca_vectores']  # noqa
scores_bptt_no_pen_all[:, 0] *= -1
scores_bptt_pen_all = bptt_pen['all_transformed_pca_vectores']  # noqa
scores_bptt_pen_all[:, 0] *= -1
scores_cbp_all = cbp['all_transformed_pca_vectores']  # noqa
scores_cbp_all[:, 0] *= -1
scores_bptt_no_pen_T = bptt_no_pen['all_transformed_pca_vectores'][bptt_no_pen['timesteps'] < 6]  # noqa
scores_bptt_pen_T = bptt_pen['all_transformed_pca_vectores'][bptt_pen['timesteps'] < 6]  # noqa
scores_cbp_T = cbp['all_transformed_pca_vectores'][cbp['timesteps'] < 20]  # noqa
scores_bptt_no_pen_TD = bptt_no_pen['all_transformed_pca_vectores'][bptt_no_pen['timesteps'] >= 6]  # noqa
scores_bptt_pen_TD = bptt_pen['all_transformed_pca_vectores'][bptt_pen['timesteps'] >= 6]  # noqa
scores_cbp_TD = cbp['all_transformed_pca_vectores'][cbp['timesteps'] >= 20]  # noqa
num_no_pen = (bptt_no_pen['timesteps'] == 1).sum()
num_pen = (bptt_pen['timesteps'] == 1).sum()
num_cbp = (cbp['timesteps'] == 1).sum()
idx_no_pen = np.arange(num_no_pen).reshape(1, -1).repeat(len(bptt_no_pen['timesteps']) / num_no_pen, 0).reshape(-1, 1)  # noqa
idx_pen = np.arange(num_pen).reshape(1, -1).repeat(len(bptt_pen['timesteps']) / num_pen, 0).reshape(-1, 1)  # noqa
idx_cbp = np.arange(num_cbp).reshape(1, -1).repeat(len(cbp['timesteps']) / num_cbp, 0).reshape(-1, 1)  # noqa

# Plot examples
fig = plt.figure(figsize=(10, 5))
plt.subplot(221)
plt.imshow(cbp['input_image'].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(222)
plt.imshow(cbp['groundtruth'].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(223)
plt.imshow(cbp['input_image'].squeeze(), cmap='Greys')
plt.axis('off')
plt.subplot(224)
plt.imshow(cbp['groundtruth'].squeeze(), cmap='Greys')
plt.axis('off')
plt.show()
plt.close(fig)

# sel_ims = np.array([1, 6, 20, 39])
fig = plt.figure(figsize=(10, 10))
plt.subplot(231)
plt.imshow(np.array(bptt_no_pen['readout'])[7].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(232)
plt.imshow(np.array(bptt_pen['readout'])[7].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(233)
plt.imshow(np.array(cbp['readout'])[-10].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(234)
plt.imshow(np.array(bptt_no_pen['readout'])[-1].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(235)
plt.imshow(np.array(bptt_pen['readout'])[-1].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(236)
plt.imshow(np.array(cbp['readout'])[-1].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.show()
plt.close(fig)

# Make timecourse plots
cmap = 'gist_heat'
interpolation = 'spline36'
start_bptt_no_pen = colored_pf(data=bptt_no_pen, start_ts=0, end_ts=7)
end_bptt_no_pen = colored_pf(data=bptt_no_pen, start_ts=7, end_ts=len(bptt_no_pen['readout']))  # noqa
start_bptt_pen = colored_pf(data=bptt_pen, start_ts=0, end_ts=7)
end_bptt_pen = colored_pf(data=bptt_pen, start_ts=7, end_ts=len(bptt_pen['readout']))  # noqa
start_cbp = colored_pf(data=cbp, start_ts=0, end_ts=30)
end_cbp = colored_pf(data=cbp, start_ts=25, end_ts=len(cbp['readout']))
f = plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(start_bptt_no_pen, cmap=cmap, interpolation=interpolation)
plt.axis('off')
plt.subplot(3, 2, 2)
# plt.imshow(end_bptt_no_pen, cmap=cmap, interpolation=interpolation)
plt.imshow(np.array(bptt_no_pen['readout'])[-1].squeeze(), cmap='Greys_r', interpolation=interpolation)  # noqa
plt.axis('off')
plt.subplot(3, 2, 3)
plt.imshow(start_bptt_pen, cmap=cmap, interpolation=interpolation)
plt.axis('off')
plt.subplot(3, 2, 4)
# plt.imshow(end_bptt_pen, cmap=cmap, interpolation=interpolation)
plt.imshow(np.array(bptt_pen['readout'])[-1].squeeze(), cmap='Greys_r', interpolation=interpolation)  # noqa
plt.axis('off')
plt.subplot(3, 2, 5)
plt.imshow(start_cbp, cmap=cmap, interpolation=interpolation)
plt.axis('off')
plt.subplot(3, 2, 6)
# plt.imshow(end_cbp, cmap=cmap, interpolation=interpolation)
plt.imshow(np.array(cbp['readout'])[-1].squeeze(), cmap='Greys_r', interpolation=interpolation)  # noqa
plt.axis('off')
plt.show()
plt.close(f)

# Distance histogram between T and T=-1
bptt_no_pen_dists = np.mean(
    (bptt_no_pen['all_transformed_pca_vectores'][bptt_no_pen['timesteps'] == 6] -  # noqa
        bptt_no_pen['all_transformed_pca_vectores'][bptt_no_pen['timesteps'] == 40]) ** 2, -1)  # noqa
bptt_pen_dists = np.mean(
    (bptt_pen['all_transformed_pca_vectores'][bptt_pen['timesteps'] == 6] -
        bptt_pen['all_transformed_pca_vectores'][bptt_pen['timesteps'] == 40]) ** 2, -1)  # noqa
cbp_pen_dists = np.mean(
    (cbp['all_transformed_pca_vectores'][cbp['timesteps'] == 30] -
        cbp['all_transformed_pca_vectores'][cbp['timesteps'] == 40]) ** 2, -1)
fig, ax = plt.subplots(1, 1, figsize=(5, 1))
sns.kdeplot(
    bptt_no_pen_dists,
    color='#861001',
    label='bptt_no_pen',
    shade=True,
    legend=False,
    ax=ax)
sns.kdeplot(
    bptt_pen_dists,
    color='#f18500',
    label='bptt_pen',
    shade=True,
    legend=False,
    ax=ax)
sns.kdeplot(
    cbp_pen_dists,
    color='#164F86',
    label='bptt_no_pen',
    shade=True,
    legend=False,
    ax=ax)
ax.set_ylabel('Density')
ax.set_xlabel('Euclidean distance between t=T and last timestep')
plt.show()
plt.close(fig)
stats.ks_2samp(bptt_no_pen_dists, bptt_pen_dists)
stats.ks_2samp(bptt_pen_dists, bptt_pen_dists)
stats.ks_2samp(bptt_no_pen_dists, cbp_pen_dists)
print('BPTT mu {} sd {}'.format(bptt_no_pen_dists.mean(), bptt_no_pen_dists.std()))
print('C-BPTT mu {} sd {}'.format(bptt_pen_dists.mean(), bptt_pen_dists.std()))
print('C-RBP mu {} sd {}'.format(cbp_pen_dists.mean(), cbp_pen_dists.std()))

# State space
cmap = 'Greys'
n_levels = 10
gridsize = (400, 300)
linewidths = 1.25
alpha = .8
figsize = (12, 4)
fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
trace = scores_bptt_no_pen_all[
    np.unique(bptt_no_pen['timesteps'], return_index=True)[1]]
plt.hexbin(
    scores_bptt_no_pen_all[:, 0],
    scores_bptt_no_pen_all[:, 1],
    # norm=norm,
    bins='log',
    cmap=cmap,
    gridsize=gridsize)
sns.kdeplot(
    scores_bptt_no_pen_T[:, 0],
    scores_bptt_no_pen_T[:, 1],
    n_levels=n_levels,
    cmap='Reds',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
sns.kdeplot(
    scores_bptt_no_pen_TD[:, 0],
    scores_bptt_no_pen_TD[:, 1],
    n_levels=n_levels,
    cmap='Blues',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
ax.plot(
    trace[:, 0],
    trace[:, 1],
    color='#f0c500',
    lw=2.,
    alpha=0.8,
    ls='-')
ax.set_xlabel('PC 1 ({}% variance)'.format(
    np.round(bptt_no_pen['explained_variance_ratio'][0] * 100)))
ax.set_ylabel('PC 2 ({}% variance)'.format(
    np.round(bptt_no_pen['explained_variance_ratio'][1] * 100)))
plt.show()
plt.close(fig)

fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
trace = scores_bptt_pen_all[
    np.unique(bptt_pen['timesteps'], return_index=True)[1]]
plt.hexbin(
    scores_bptt_pen_all[:, 0],
    scores_bptt_pen_all[:, 1],
    bins='log',
    cmap=cmap,
    gridsize=gridsize)
sns.kdeplot(
    scores_bptt_pen_T[:, 0],
    scores_bptt_pen_T[:, 1],
    n_levels=n_levels,
    cmap='Reds',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
sns.kdeplot(
    scores_bptt_pen_TD[:, 0],
    scores_bptt_pen_TD[:, 1],
    n_levels=n_levels,
    cmap='Blues',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
ax.plot(
    trace[:, 0],
    trace[:, 1],
    color='#f0c500',
    lw=2.,
    alpha=0.8,
    ls='-')
ax.set_xlabel('PC 1 ({}% variance)'.format(
    np.round(bptt_pen['explained_variance_ratio'][0] * 100)))
ax.set_ylabel('PC 2 ({}% variance)'.format(
    np.round(bptt_pen['explained_variance_ratio'][1] * 100)))
plt.show()
plt.close(fig)

fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
trace = scores_cbp_all[
    np.unique(cbp['timesteps'], return_index=True)[1]]
plt.hexbin(
    scores_cbp_all[:, 0],
    scores_cbp_all[:, 1],
    bins='log',
    cmap=cmap,
    gridsize=gridsize)
sns.kdeplot(
    scores_cbp_T[:, 0],
    scores_cbp_T[:, 1],
    n_levels=n_levels,
    cmap='Reds',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
sns.kdeplot(
    scores_cbp_TD[:, 0],
    scores_cbp_TD[:, 1],
    n_levels=n_levels,
    cmap='Blues',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
ax.plot(
    trace[:, 0],
    trace[:, 1],
    color='#f0c500',
    lw=2.,
    alpha=0.8,
    ls='-')
ax.set_xlabel('PC 1 ({}% variance)'.format(
    np.round(bptt_pen['explained_variance_ratio'][0] * 100)))
ax.set_ylabel('PC 2 ({}% variance)'.format(
    np.round(bptt_pen['explained_variance_ratio'][1] * 100)))
plt.show()
plt.close(fig)
