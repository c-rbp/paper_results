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
BPTT_pertimestep_hGRU = np.load('BPTT_pertimestep_hGRU.p', allow_pickle=True)
TBPTT_hGRU = np.load('TBPTT_hGRU.p', allow_pickle=True)
cbp = np.load('CBP.p', allow_pickle=True)

scores_BPTT_pertimestep_hGRU_all = BPTT_pertimestep_hGRU['all_transformed_pca_vectores']  # noqa
scores_BPTT_pertimestep_hGRU_all[:, 0] *= -1
scores_TBPTT_hGRU_all = TBPTT_hGRU['all_transformed_pca_vectores']  # noqa
scores_TBPTT_hGRU_all[:, 0] *= -1
scores_cbp_all = cbp['all_transformed_pca_vectores']  # noqa
scores_cbp_all[:, 0] *= -1
scores_BPTT_pertimestep_hGRU_T = BPTT_pertimestep_hGRU['all_transformed_pca_vectores'][BPTT_pertimestep_hGRU['timesteps'] < 6]  # noqa
scores_TBPTT_hGRU_T = TBPTT_hGRU['all_transformed_pca_vectores'][TBPTT_hGRU['timesteps'] < 6]  # noqa
scores_cbp_T = cbp['all_transformed_pca_vectores'][cbp['timesteps'] < 30]  # noqa
scores_BPTT_pertimestep_hGRU_TD = BPTT_pertimestep_hGRU['all_transformed_pca_vectores'][BPTT_pertimestep_hGRU['timesteps'] >= 6]  # noqa
scores_TBPTT_hGRU_TD = TBPTT_hGRU['all_transformed_pca_vectores'][TBPTT_hGRU['timesteps'] >= 6]  # noqa
scores_cbp_TD = cbp['all_transformed_pca_vectores'][cbp['timesteps'] >= 30]  # noqa
num_no_pen = (BPTT_pertimestep_hGRU['timesteps'] == 1).sum()
num_pen = (TBPTT_hGRU['timesteps'] == 1).sum()
num_cbp = (cbp['timesteps'] == 1).sum()
idx_no_pen = np.arange(num_no_pen).reshape(1, -1).repeat(len(BPTT_pertimestep_hGRU['timesteps']) / num_no_pen, 0).reshape(-1, 1)  # noqa
idx_pen = np.arange(num_pen).reshape(1, -1).repeat(len(TBPTT_hGRU['timesteps']) / num_pen, 0).reshape(-1, 1)  # noqa
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
plt.imshow(np.array(BPTT_pertimestep_hGRU['readout'])[7].squeeze(), cmap='Greys_r')  # noqa
plt.axis('off')
plt.subplot(232)
plt.imshow(np.array(TBPTT_hGRU['readout'])[7].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(233)
plt.imshow(np.array(cbp['readout'])[-10].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(234)
plt.imshow(np.array(BPTT_pertimestep_hGRU['readout'])[-1].squeeze(), cmap='Greys_r')  # noqa
plt.axis('off')
plt.subplot(235)
plt.imshow(np.array(TBPTT_hGRU['readout'])[-1].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.subplot(236)
plt.imshow(np.array(cbp['readout'])[-1].squeeze(), cmap='Greys_r')
plt.axis('off')
plt.show()
plt.close(fig)

# Make timecourse plots
cmap = 'gist_heat'
interpolation = 'spline36'
start_bptt_no_pen = colored_pf(data=BPTT_pertimestep_hGRU, start_ts=0, end_ts=7)
end_bptt_no_pen = colored_pf(data=BPTT_pertimestep_hGRU, start_ts=7, end_ts=len(BPTT_pertimestep_hGRU['readout']))  # noqa
start_bptt_pen = colored_pf(data=TBPTT_hGRU, start_ts=0, end_ts=7)
end_bptt_pen = colored_pf(data=TBPTT_hGRU, start_ts=7, end_ts=len(TBPTT_hGRU['readout']))  # noqa
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
BPTT_pertimestep_hGRU_dists = np.mean(
    (BPTT_pertimestep_hGRU['all_transformed_pca_vectores'][BPTT_pertimestep_hGRU['timesteps'] == 6] -  # noqa
        BPTT_pertimestep_hGRU['all_transformed_pca_vectores'][BPTT_pertimestep_hGRU['timesteps'] == 40]) ** 2, -1)  # noqa
TBPTT_hGRU_dists = np.mean(
    (TBPTT_hGRU['all_transformed_pca_vectores'][TBPTT_hGRU['timesteps'] == 6] -
        TBPTT_hGRU['all_transformed_pca_vectores'][TBPTT_hGRU['timesteps'] == 40]) ** 2, -1)  # noqa
cbp_pen_dists = np.mean(
    (cbp['all_transformed_pca_vectores'][cbp['timesteps'] == 30] -
        cbp['all_transformed_pca_vectores'][cbp['timesteps'] == 40]) ** 2, -1)
fig, ax = plt.subplots(1, 1, figsize=(5, 1))
sns.kdeplot(
    BPTT_pertimestep_hGRU_dists,
    color='#861001',
    label='BPTT_pertimestep_hGRU',
    shade=True,
    legend=False,
    ax=ax)
sns.kdeplot(
    TBPTT_hGRU_dists,
    color='#f18500',
    label='TBPTT_hGRU',
    shade=True,
    legend=False,
    ax=ax)
sns.kdeplot(
    cbp_pen_dists,
    color='#164F86',
    label='BPTT_pertimestep_hGRU',
    shade=True,
    legend=False,
    ax=ax)
ax.set_ylabel('Density')
ax.set_xlabel('Euclidean distance between t=T and last timestep')
plt.show()
plt.close(fig)
print(stats.ks_2samp(BPTT_pertimestep_hGRU_dists, TBPTT_hGRU_dists))
print(stats.ks_2samp(TBPTT_hGRU_dists, TBPTT_hGRU_dists))
print(stats.ks_2samp(BPTT_pertimestep_hGRU_dists, cbp_pen_dists))

# State space
cmap = 'Greys'
n_levels = 10
gridsize = (400, 300)
linewidths = 1.25
alpha = .8
figsize = (12, 4)
fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
trace = scores_BPTT_pertimestep_hGRU_all[
    np.unique(BPTT_pertimestep_hGRU['timesteps'], return_index=True)[1]]
plt.hexbin(
    scores_BPTT_pertimestep_hGRU_all[:, 0],
    scores_BPTT_pertimestep_hGRU_all[:, 1],
    # norm=norm,
    bins='log',
    cmap=cmap,
    gridsize=gridsize)
sns.kdeplot(
    scores_BPTT_pertimestep_hGRU_T[:, 0],
    scores_BPTT_pertimestep_hGRU_T[:, 1],
    n_levels=n_levels,
    cmap='Reds',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
sns.kdeplot(
    scores_BPTT_pertimestep_hGRU_TD[:, 0],
    scores_BPTT_pertimestep_hGRU_TD[:, 1],
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
    np.round(BPTT_pertimestep_hGRU['explained_variance_ratio'][0] * 100)))
ax.set_ylabel('PC 2 ({}% variance)'.format(
    np.round(BPTT_pertimestep_hGRU['explained_variance_ratio'][1] * 100)))
plt.show()
plt.close(fig)

fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
trace = scores_TBPTT_hGRU_all[
    np.unique(TBPTT_hGRU['timesteps'], return_index=True)[1]]
plt.hexbin(
    scores_TBPTT_hGRU_all[:, 0],
    scores_TBPTT_hGRU_all[:, 1],
    bins='log',
    cmap=cmap,
    gridsize=gridsize)
sns.kdeplot(
    scores_TBPTT_hGRU_T[:, 0],
    scores_TBPTT_hGRU_T[:, 1],
    n_levels=n_levels,
    cmap='Reds',
    linewidths=linewidths,
    alpha=alpha,
    ax=ax)
sns.kdeplot(
    scores_TBPTT_hGRU_TD[:, 0],
    scores_TBPTT_hGRU_TD[:, 1],
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
    np.round(TBPTT_hGRU['explained_variance_ratio'][0] * 100)))
ax.set_ylabel('PC 2 ({}% variance)'.format(
    np.round(TBPTT_hGRU['explained_variance_ratio'][1] * 100)))
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
    np.round(TBPTT_hGRU['explained_variance_ratio'][0] * 100)))
ax.set_ylabel('PC 2 ({}% variance)'.format(
    np.round(TBPTT_hGRU['explained_variance_ratio'][1] * 100)))
plt.show()
plt.close(fig)
