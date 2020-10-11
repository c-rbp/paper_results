
import numpy as np
import pandas as pd



baseline = pd.read_json('scratch-RN50-1x-baseline-test1/inference/coco_instances_results.json')
cbp = pd.read_json('RFPN_cbp20-test/inference/coco_instances_results.json')
unique_cbp_images = np.unique(cbp['image_id'])
cbp_image_score = {}
for im in unique_cbp_images:
   scores = cbp[cbp['image_id'] == im].score
   scores = scores[:5]
   cbp_image_score[im] = scores.mean()


unique_baseline_images = np.unique(baseline['image_id'])
baseline_image_score = {}
for im in unique_baseline_images:
   scores = baseline[baseline['image_id'] == im].score
   scores = scores[:5]
   baseline_image_score[im] = scores.mean()

diffs = []
for im in unique_baseline_images:
    diffs += [cbp_image_score[im] - baseline_image_score[im]]
diffs = np.array(diffs)
nan_diffs = np.isnan(diffs)
diffs = diffs[~nan_diffs]
unique_baseline_images = unique_baseline_images[~nan_diffs]
idx = np.argsort(diffs)[::-1]
np.save('top_100', unique_baseline_images[idx[:100]])
# np.save('all_diffs', unique_baseline_images[idx[:100]])



import numpy as np


cbp = np.load('coco_inst/cbp_coco.npz', allow_pickle=True)
baseline = np.load('coco_inst/baseline_coco.npz', allow_pickle=True)
cbp_ious = cbp['ious'].item()
cbp_iou_idx = np.array([x[0] for x in cbp_ious.keys()])
cbp_ious = np.array([x for x in cbp_ious.values()])
cbp_evals = cbp['evalimgs']
cbp_imgs = np.unique(cbp['imgids'])
cbp_imgs_ious = cbp_imgs.reshape(1, -1).repeat(4)
cbp_imgs_evals = cbp_imgs.reshape(1, -1).repeat(80 * 4)

baseline_ious = baseline['ious'].item()
baseline_iou_idx = np.array([x[0] for x in baseline_ious.keys()])
baseline_ious = np.array([x for x in baseline_ious.values()])
baseline_evals = baseline['evalimgs']
baseline_imgs = np.unique(baseline['imgids'])
baseline_imgs_ious = baseline_imgs.reshape(1, -1).repeat(4)
baseline_imgs_evals = baseline_imgs.reshape(1, -1).repeat(80 * 4)

cbp_image_score = {}
for im in cbp_imgs:
   idx = cbp_iou_idx == im
   eval_list = cbp_ious[idx]
   evals = []
   for e in eval_list:
      if len(e):
         evals.append(e[np.nonzero(e)].mean())
   cbp_image_score[im] = np.array(evals).mean()

baseline_image_score = {}
for im in baseline_imgs:
   idx = baseline_iou_idx == im
   eval_list = baseline_ious[idx]
   evals = []
   for e in eval_list:
      if len(e):
         evals.append(e[np.nonzero(e)].mean())
   baseline_image_score[im] = np.array(evals).mean()


diffs = {}
for im in baseline_imgs:
   idx = baseline_iou_idx == im
   baseline_eval_list = baseline_ious[idx]
   cbp_eval_list = cbp_ious[idx]
   evals = []
   for ba, cb in zip(baseline_eval_list, cbp_eval_list):
      if len(ba) and len(cb):
         d = cb - ba
         d = d[np.nonzero(d)]
         evals.append(d.mean())
   diffs[im] = np.array(evals).mean()
cbp_imgs = diffs.keys()
diffs = np.array(list(diffs.values()))
cbp_imgs = np.array(list(cbp_imgs))
nan_diffs = np.isnan(diffs)
diffs = diffs[~nan_diffs]
cbp_imgs = cbp_imgs[~nan_diffs]
idx = np.argsort(diffs)[::-1]
np.save('coco_inst/top_100', cbp_imgs[idx[:100]])



diffs = []
cbp_imgs = []
for cb, ba in zip(cbp_evals, baseline_evals):
   if cb is not None and ba is not None:
      # d = np.array(cb['dtScores']).mean() / np.array(cb['dtScores']).std() - np.array(ba['dtScores']).mean() / np.array(ba['dtScores']).std()
      if len(cb['dtScores']) and len(ba['dtScores']):
         d = np.array(cb['dtScores']).max() - np.array(ba['dtScores']).max()
         # d = d[np.nonzero(d)]
         diffs.append(d)
         cbp_imgs.append(cb['image_id'])
cbp_imgs = np.array(cbp_imgs)
diffs = np.array(diffs)
nan_diffs = np.isnan(diffs)
diffs = diffs[~nan_diffs]
cbp_imgs = cbp_imgs[~nan_diffs]
idx = np.argsort(diffs)[::-1]
np.save('coco_inst/top_100', cbp_imgs[idx[:100]])






diffs = []
for im in cbp_imgs:
    diffs += [cbp_image_score[im] - baseline_image_score[im]]
diffs = np.array(diffs)
nan_diffs = np.isnan(diffs)
diffs = diffs[~nan_diffs]
cbp_imgs = cbp_imgs[~nan_diffs]
idx = np.argsort(diffs)[::-1]
np.save('coco_inst/top_100', cbp_imgs[idx[:100]])




import os
import numpy as np
from glob import glob
from skimage import io as misc
from matplotlib import pyplot as plt
f0 = 'RN50-baseline'
f1 = 'visualizations/CBP20_tau_0.9'
f0f = glob(os.path.join(f0, '*'))
f1f = glob(os.path.join(f1, '*'))
idxs = np.load('coco_inst/top_100.npy')
# for i0, i1 in zip(f0f, f1f):
#    fig = plt.figure(figsize=(12, 8))
#    plt.suptitle(i0.split(os.path.sep)[1])
#    plt.subplot(121)
#    plt.imshow(misc.imread(i0))
#    plt.title(i0.split(os.path.sep)[0])
#    plt.axis('off')
#    plt.subplot(122)
#    plt.imshow(misc.imread(i1))
#    plt.axis('off')
#    plt.title(i1.split(os.path.sep)[0])
#    plt.show()
#    plt.close(fig)

for idx in idxs:
   i0_im = os.path.join(f0, '{}.jpg'.format(idx))
   fig = plt.figure(figsize=(12, 8))
   plt.suptitle(idx)
   plt.subplot(121)
   plt.imshow(misc.imread(i0_im))
   plt.title(f0)
   plt.axis('off')
   plt.subplot(122)
   i1_im = os.path.join(f1, '{}.jpg'.format(idx))
   plt.imshow(misc.imread(i1_im))
   plt.axis('off')
   plt.title(f1)
   plt.show()
   plt.close(fig)

