import os
import numpy as np
import seaborn as sns
import pandas as pd  # noqa
from matplotlib import pyplot as plt


"""

1. 

There is growing consensus that visual routines depend on recurrent processing,
from core object recognition to incremental grouping. These findings are consistent
with a number of reports suggesting that recurrent neural networks (RNNs) are better
at learning small-scale visual tasks and explaining neural responses in primate
visual cortex than feedforward neural networks (FNNs). In light of the great promise
of recurrent processing, why are large-scale computer vision benchmarks dominated by FNNs?

We posit that RNNs are limited by standard Backpropagation Through Time (BPTT) training,
which costs $O(N)$ memory to train a $N$ time step model, making it infeasible to develop
RNNs that rival the extraordinary capacity of state-of-the-art FNNs. We address this
problem with Constrained Backpropagation (CBP), a learning rule which leverages
auto-differentiation to guarantee dissipative dynamics and achieves $O(1)$
memory-complexity. We first find that this efficiency means that CBP-trained
RNN can learn long-range spatial dependencies in a synthetic contour tracing task that
BPTT-trained RNNs cannot. Next, we extend CBP to the large-scale MS-COCO challenge, and
show that a CBP-trained recurrent FPN-Mask-RCNN outperforms the state-of-the-art in
Panoptic segmentation. CBP is a general purpose algorithm for any computational domain
where complex recurrent processing explains the brain or behavior. We make our code and
datasets publicly available.

2. "Are rnns trained for visual tasks stable?": there are cheap ways of training RNNs, but these depend on "stability" (show equation).
   a. Related work on fixed point optim and RBP
   b. Only explain the BPTT results. Performance monotonically decreases when T' > T.
   c. Here we explain why...
3. Method with proof
4. PF results
    a. Logic (maybe move to results):
    Different images take different Ts (difficulty)
    Training maybe does not sample all Ts (generalization)
    We suspect that this forces feedforward-esque recurrent solutions.
    If this is not the case, then a model trained for T should be able to smoothly extrapolate to T+D timesteps.
5. Panoptic results
6. Related works
    a. Neural ODE
    b. DEQ
    c. Chelsey finn stuff
    d. Kernel RNN/eligibility traces
7. Discussion


"""

# Ideal results
# Fig 0: State space (extrapolation)
# Fig 1a: Extrapolation performance for BPTT-trained hgru and lstm
# Fig 1b: Extrapolation performance for BPTT + Pen hgru and lstm
# Fig 2: Stability expressivity tradeoff
# Fig 3a: PF Cliff generalization (alekh is finishing this)
# Fig 3b: PF Cliff extrapolation (alekh is finishing this)
# Fig 4: Solving PF with cheaper models but more TS
# Fig 5: Panoptic results with resnet 50 + maybe resnet 101
# Fig 6: Timesteps of processing on panoptic




# Fig 1a: Extrapolation performance for BPTT-trained hgru and lstm  # noqa



# # Fig 3: Pathfinder cliff
def f_to_iou(f):  # noqa
    """Convert f score to iou."""
    return f / (2. - f)

# Generalization
pf14_bptt_pen = np.load(os.path.join('BPTT_penalty_hgru_PF14SEG_6ts_4GPU_128Batch', 'val.npz'))  # noqa
pf20_bptt_pen = np.load(os.path.join('BPTT_penalty_hgru_PF20SEG_6ts_4GPU_128Batch', 'val.npz'))  # noqa
pf14_bptt_nopen = np.load(os.path.join('BPTT_nopen_hgru_PF14SEG_6ts_4GPU_128Batch_P4', 'val.npz'))  # noqa
pf20_bptt_nopen = np.load(os.path.join('BPTT_nopen_hgru_PF20SEG_6ts_4GPU_128Batch_P4', 'val.npz'))  # noqa
pf14_cbp = np.load(os.path.join('RBP_penalty_hgru_PF14SEG_20ts_15iter_4GPU_128Batch_P4', 'val.npz'))  # noqa
pf20_cbp = np.load(os.path.join('RBP_penalty_hgru_PF20SEG_30ts_15iter_4GPU_128Batch', 'val.npz'))  # noqa
pf14_rbp = np.load(os.path.join('RBP_nopen_hgru_PF14SEG_20ts_20iter_4GPU_128Batch_P5', 'val.npz'))  # noqa
pf20_rbp = np.load(os.path.join('RBP_nopen_hgru_PF20SEG_20ts_15iter_4GPU_128Batch_P5', 'val.npz'))  # noqa

pf14_bptt_pen_f = f_to_iou(pf14_bptt_pen['f1score'].max())
pf20_bptt_pen_f = f_to_iou(pf20_bptt_pen['f1score'].max())
pf14_bptt_nopen_f = f_to_iou(pf14_bptt_nopen['f1score'].max())
pf20_bptt_nopen_f = f_to_iou(pf20_bptt_nopen['f1score'].max())
pf14_cbp_f = f_to_iou(pf14_cbp['f1score'].max())
pf20_cbp_f = f_to_iou(pf20_cbp['f1score'].max())
pf14_rbp_f = f_to_iou(pf14_rbp['f1score'].max())
pf20_rbp_f = f_to_iou(pf20_rbp['f1score'].max())
df = pd.DataFrame(
    np.vstack((
        np.stack((
            pf14_bptt_pen_f,
            pf20_bptt_pen_f,
            pf14_bptt_nopen_f,
            pf20_bptt_nopen_f,
            pf14_rbp_f,
            pf20_rbp_f,
            pf14_cbp_f,
            pf20_cbp_f,
        )),
        np.stack((
            '14',
            '20',
            '14',
            '20',
            '14',
            '20',
            '14',
            '20',
        )),
        np.stack((
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'RBP',
            'RBP',
            'CBP',
            'CBP',
        )),
        np.stack((
            '#ad0000',
            '#ad0000',
            '#962b00',
            '#962b00',
            '#a632a8',
            '#a632a8',
            '#004099',
            '#004099',
        )),
        np.stack((
            '-',
            '-',
            '--',
            '--',
            '--',
            '--',
            '-',
            '-',
        )),
    )).transpose(),
    columns=['IOU', 'PF-length', 'Model', 'color', 'style'])
df.IOU = pd.to_numeric(df.IOU)
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df,
    y='IOU',
    x='PF-length',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    ax=ax1)
box = g.get_position()
g.set_position([box.x0, box.y0, box.width * 0.65, box.height])
g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
plt.title('Training/testing on the same PF dataset.')
plt.show()
plt.close(fig)

# Extrapolation
cbp14to20 = '0.         0.13939447 0.19670977 0.2221724  0.29016967 0.3204802\
 0.43542857 0.49493094 0.5717055  0.62979887 0.68352951 0.72074954\
 0.76058064 0.7977819  0.83360763 0.86793519 0.89219576 0.91726667\
 0.9327627  0.94452303 0.9568256  0.96390257 0.9678023  0.97020593\
 0.97083278 0.9727167  0.97337605 0.97409519 0.97561815 0.97539362\
 0.97442399 0.97453431 0.97445854 0.9747562  0.97427007 0.97504883\
 0.97429221 0.97413516 0.9727169  0.97339339 0.97349878'
cbp14to20 = cbp14to20.split()
cbp14to20 = list(map(float, cbp14to20))
cbp14to14 = '0.         0.1829926  0.25934524 0.28414221 0.34207375 0.38730507\
 0.51196631 0.58164009 0.67349069 0.73279841 0.7958541  0.84236955\
 0.8945225  0.92225528 0.9437502  0.95667498 0.96292194 0.9688264\
 0.97395582 0.97555502 0.97566584 0.97748717 0.97769083 0.97705198\
 0.97495628 0.97827868 0.97698296 0.9772331  0.97692297 0.9768163\
 0.97507079 0.97674918 0.97504884 0.97436971 0.97342388 0.97249485\
 0.97184831 0.97088757 0.97196722 0.97159534 0.97071631'
cbp14to14 = cbp14to14.split()
cbp14to14 = list(map(float, cbp14to14))
nopenbptt14to14 = '0.00000000e+00 9.05797118e-04 2.25504794e-01 4.18992484e-01\
 5.59248881e-01 6.79815400e-01 7.66749477e-01 8.42635058e-01\
 9.15523475e-01 9.76441046e-01 9.90833955e-01 9.87808295e-01\
 9.80914518e-01 9.61729631e-01 9.30203671e-01 8.76480627e-01\
 7.71850900e-01 6.39693829e-01 5.39056032e-01 4.68481390e-01\
 4.00991427e-01 3.42105599e-01 3.06715640e-01 2.84520344e-01\
 2.74259796e-01 2.64779816e-01 2.58281865e-01 2.53176205e-01\
 2.44417488e-01 2.42290147e-01 2.39064494e-01 2.36233083e-01\
 2.34455935e-01 2.29960646e-01 2.25215944e-01 2.19440059e-01\
 2.13491274e-01 2.04977590e-01 1.98315753e-01 1.91847100e-01\
 1.85771846e-01'
nopenbptt14to14 = nopenbptt14to14.split()
nopenbptt14to14 = list(map(float, nopenbptt14to14))
nopenbptt14to20 = '0.         0.00667139 0.18217864 0.32913517 0.45975179 0.56879125\
 0.64489034 0.71070009 0.77129525 0.83699756 0.87885363 0.90721921\
 0.93026165 0.94196481 0.91346009 0.83899726 0.71998586 0.59055702\
 0.48771953 0.41409081 0.34632249 0.29586801 0.26267421 0.24204779\
 0.22277666 0.2134171  0.20733075 0.20149788 0.19935373 0.19720987\
 0.19555713 0.19371332 0.19390638 0.19250777 0.19061989 0.18493876\
 0.18079081 0.17935898 0.17742869 0.17203339 0.16791184'
nopenbptt14to20 = nopenbptt14to20.split()
nopenbptt14to20 = list(map(float, nopenbptt14to20))
wpenbptt14to14 = '0.         0.28911894 0.56361696 0.69754296 0.81410841 0.92092348\
 0.99364966 0.98690056 0.98608936 0.98537767 0.98484941 0.9757118\
 0.96480662 0.95073899 0.9429411  0.93910663 0.9351619  0.93066135\
 0.91786137 0.90631741 0.90168027 0.89641625 0.89261776 0.88998553\
 0.8857479  0.88441861 0.8831189  0.88039889 0.87701352 0.87759928\
 0.87523622 0.87480201 0.87366332 0.87187211 0.87097425 0.86929272\
 0.86779315 0.86479829 0.86275955 0.85956575 0.85556371'
wpenbptt14to14 = wpenbptt14to14.split()
wpenbptt14to14 = list(map(float, wpenbptt14to14))
wpenbptt14to20 = '0.         0.21912178 0.49530698 0.56487864 0.67427731 0.76940883\
 0.85683353 0.9316571  0.98129999 0.98872717 0.98446153 0.97558048\
 0.9360606  0.90727764 0.88406001 0.86782136 0.86076783 0.86161605\
 0.86805108 0.86803956 0.86354205 0.85524569 0.84269934 0.8343066\
 0.82768124 0.82446647 0.82370602 0.82291713 0.82180925 0.8195186\
 0.81636456 0.81079077 0.80784607 0.80795143 0.81005309 0.80970154\
 0.80885082 0.80875186 0.80784734 0.80642607 0.80322549'
wpenbptt14to20 = wpenbptt14to20.split()
wpenbptt14to20 = list(map(float, wpenbptt14to20))
df = pd.DataFrame(
    np.vstack((
        np.stack((
            cbp14to14,
            cbp14to20,
            nopenbptt14to14,
            nopenbptt14to20,
            wpenbptt14to14,
            wpenbptt14to20), -1
        ),
        np.stack((
            '14',
            '20',
            '14',
            '20',
            '14',
            '20',
        )),
        np.stack((
            'CBP',
            'CBP',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'BPTT w/ penalty',
            'BPTT w/ penalty',
        )),
        np.stack((
            '#004099',
            '#004099',
            '#ad0000',
            '#ad0000',
            '#962b00',
            '#962b00',
        )),
        np.stack((
            '-',
            '-',
            '--',
            '--',
            '-',
            '-',
        )),
    )).transpose(),
    columns=['IOU', 'PF-length', 'Model', 'color', 'style'])
df.IOU = pd.to_numeric(df.IOU)
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df,
    y='IOU',
    x='PF-length',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    ax=ax1)
box = g.get_position()
g.set_position([box.x0, box.y0, box.width * 0.65, box.height])
g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
plt.title('Training/testing on the same PF dataset.')
plt.show()
plt.close(fig)





# Examples