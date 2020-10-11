import os
import torch
import numpy as np
import seaborn as sns
import pandas as pd  # noqa
from matplotlib import pyplot as plt
from skimage import io


"""
Ideal results
Fig 0: State space (extrapolation)
Fig 1a: Extrapolation performance for BPTT-trained hgru and lstm
Fig 1b: Extrapolation performance for BPTT + Pen hgru and lstm
Fig 2: Stability expressivity tradeoff
Fig 3a: PF Cliff generalization (alekh is finishing this)
Fig 3b: PF Cliff extrapolation (alekh is finishing this)
Fig 4: Solving PF with cheaper models but more TS
Fig 5: Panoptic results with resnet 50 + maybe resnet 101
Fig 6: Timesteps of processing on panoptic
"""


def f_to_iou(f):  # noqa
    """Convert f score to iou."""
    return f / (2. - f)


colors = {
    'bptt_w_penalty': '#eb7100',
    'bptt_wo_penalty': '#ad0000',
    'bptt_wo_penalty_truncated': '#00941e',
    'bptt_wo_penalty_allsup': '#ba3872',
    'cbp': '#004099',
    'rbp': '#a632a8',
    'ff6': '#9c9c9c',
    'lstm_cbp': '#cc7810',
    'lstm_bptt_wo_penalty': '#c6cc10',
}
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Fig: Train/test performance on PF  # noqa
maxvalf1 = {
    'pf14_bptt_pen': 0.9918,
    'pf20_bptt_pen': 0.9144,
    'pf25_bptt_pen': 0.8469,
    'pf14_bptt_nopen': 0.9931,
    'pf20_bptt_nopen': 0.9395,
    'pf25_bptt_nopen': 0.8402,
    'pf14_rbp': 0.8332,
    'pf20_rbp': 0.7690,
    'pf25_rbp': 0.7424,
    'pf14_cbp': 0.9804,
    'pf20_cbp': 0.9725,
    'pf25_cbp': 0.9679,
    'pf14_ff6': 0.8825,
    'pf20_ff6': 0.7765,
    'pf25_ff6': 0.7208,
    'pf14_bptt_nopen_allsup': 0.9896,
    'pf20_bptt_nopen_allsup': 0.9325,
    'pf25_bptt_nopen_allsup': 0.8385,
    'pf14_bptt_nopen_trunc': 0.9569,
    'pf20_bptt_nopen_trunc': 0.8506,
    'pf25_bptt_nopen_trunc': 0.7613,
    'pf14_lstm_bptt_nopen': 0.8944,
    'pf14_cbp': 0.9653,
    'pf14_rbp': 0.8466,
}


df = pd.DataFrame(
    np.vstack((
        np.stack((
            maxvalf1['pf14_bptt_pen'],
            maxvalf1['pf20_bptt_pen'],
            maxvalf1['pf25_bptt_pen'],
            maxvalf1['pf14_bptt_nopen'],
            maxvalf1['pf20_bptt_nopen'],
            maxvalf1['pf25_bptt_nopen'],
            maxvalf1['pf14_rbp'],
            maxvalf1['pf20_rbp'],
            maxvalf1['pf25_rbp'],
            maxvalf1['pf14_cbp'],
            maxvalf1['pf20_cbp'],
            maxvalf1['pf25_cbp'],
            maxvalf1['pf14_ff6'],
            maxvalf1['pf20_ff6'],
            maxvalf1['pf25_ff6'],
        )),
        np.stack((
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
        )),
        np.stack((
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'RBP',
            'RBP',
            'RBP',
            'CBP',
            'CBP',
            'CBP',
            'FF6',
            'FF6',
            'FF6',
        )),
        np.stack((
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['rbp'],
            colors['rbp'],
            colors['rbp'],
            colors['cbp'],
            colors['cbp'],
            colors['cbp'],
            colors['ff6'],
            colors['ff6'],
            colors['ff6'],
        )),
        np.stack((
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        )),
    )).transpose(),
    columns=['IOU', 'PF-length', 'Model', 'color', 'style'])
df.IOU = f_to_iou(pd.to_numeric(df.IOU))
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
g = sns.lineplot(
    data=df,
    y='IOU',
    x='PF-length',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1.,
    ax=ax1)
box = g.get_position()
ax1.set_ylim(0.49, 1.01)
g.set_position([box.x0, box.y0, box.width * 0.65, box.height])
g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
plt.title('Training/testing on the same PF dataset.')
plt.show()
plt.close(fig)

# SI version
df = pd.DataFrame(
    np.vstack((
        np.stack((
            maxvalf1['pf14_bptt_pen'],
            maxvalf1['pf20_bptt_pen'],
            maxvalf1['pf25_bptt_pen'],
            maxvalf1['pf14_bptt_nopen_allsup'],
            maxvalf1['pf20_bptt_nopen_allsup'],
            maxvalf1['pf25_bptt_nopen_allsup'],
            maxvalf1['pf14_bptt_nopen_trunc'],
            maxvalf1['pf20_bptt_nopen_trunc'],
            maxvalf1['pf25_bptt_nopen_trunc'],
        )),
        np.stack((
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
        )),
        np.stack((
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/o penalty timestep supervision',
            'BPTT w/o penalty timestep supervision',
            'BPTT w/o penalty timestep supervision',
            'BPTT w/o penalty truncated',
            'BPTT w/o penalty truncated',
            'BPTT w/o penalty truncated',
        )),
        np.stack((
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_wo_penalty_allsup'],
            colors['bptt_wo_penalty_allsup'],
            colors['bptt_wo_penalty_allsup'],
            colors['bptt_wo_penalty_truncated'],
            colors['bptt_wo_penalty_truncated'],
            colors['bptt_wo_penalty_truncated'],
        )),
        np.stack((
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        )),
    )).transpose(),
    columns=['IOU', 'PF-length', 'Model', 'color', 'style'])
df.IOU = f_to_iou(pd.to_numeric(df.IOU))
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
g = sns.lineplot(
    data=df,
    y='IOU',
    x='PF-length',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1.,
    ax=ax1)
box = g.get_position()
ax1.set_ylim(0.49, 1.01)
g.set_position([box.x0, box.y0, box.width * 0.65, box.height])
g.legend(loc='upper right', bbox_to_anchor=(2., 0.5), ncol=1)
plt.title('Training/testing on the same PF dataset.')
plt.show()
plt.close(fig)


# Extrapolation
cbp14to14 = '0.         0.18441663 0.26982303 0.2939116  0.37880287 0.40762725\
 0.52843622 0.60180503 0.68912836 0.75248604 0.81603082 0.86235014\
 0.90707447 0.93553605 0.95617541 0.96911743 0.97234752 0.97892851\
 0.98118958 0.98455813 0.98572479 0.98659945 0.98675781 0.9867273\
 0.98692003 0.98646097 0.98650578 0.98536448 0.9854303  0.98420405\
 0.98510703 0.98290645 0.98307233 0.98223893 0.98291923 0.98252141\
 0.9829398  0.98240863 0.98169915 0.982232   0.98188494 0.98217789\
 0.98173038 0.98180267 0.98108899 0.98148434 0.98092602 0.98132334\
 0.98062066 0.98119918 0.97949886'
cbp14to14 = cbp14to14.split()
cbp14to14 = list(map(float, cbp14to14))
cbp14to20 = '0.         0.13887579 0.20020969 0.22842782 0.31183209 0.34979671\
 0.46839987 0.52891376 0.60458551 0.65686293 0.71151294 0.74971419\
 0.79104387 0.82527716 0.85737363 0.88808547 0.91499431 0.93689517\
 0.95042145 0.96292952 0.9670834  0.97080286 0.97306901 0.97426319\
 0.97536089 0.97567564 0.97540023 0.97499318 0.97429729 0.97355443\
 0.9736025  0.97303366 0.97334665 0.97234223 0.97296397 0.97210842\
 0.97223481 0.9716629  0.97120001 0.97267228 0.9718355  0.97278126\
 0.97214323 0.97170863 0.97099737 0.97136628 0.97128636 0.97131493\
 0.97198683 0.97112308 0.97160461'
cbp14to20 = cbp14to20.split()
cbp14to20 = list(map(float, cbp14to20))
cbp14to25 = '0.         0.10679946 0.16534884 0.1676654  0.18674002 0.19982644\
 0.2519543  0.29987106 0.3712354  0.41738916 0.47022    0.50864515\
 0.55145287 0.58144107 0.61162037 0.64526002 0.66628155 0.69672989\
 0.71483636 0.74085306 0.75549176 0.7784192  0.79436547 0.81102633\
 0.82211805 0.83253372 0.8451341  0.85181341 0.86200515 0.86553896\
 0.87443958 0.8755387  0.87971884 0.88161694 0.88343602 0.88366338\
 0.88656123 0.88485373 0.88554264 0.8861262  0.886631   0.88795309\
 0.88847628 0.88931444 0.88874466 0.88926563 0.88678532 0.88432592\
 0.88456822 0.88124563 0.88110512 0.88196152 0.88054927 0.88153812\
 0.88217683 0.87934609 0.88086058 0.88170401 0.88228286 0.88163025\
 0.8816744'
cbp14to25 = cbp14to25.split()
cbp14to25 = list(map(float, cbp14to25))

nopenbptt14to14 = '0.         0.16194975 0.48172862 0.71009754 0.83558101 0.92878309\
 0.99072864 0.98386084 0.95918984 0.88066249 0.74507685 0.60764117\
 0.49412411 0.41159769 0.3428012  0.300084   0.27139112 0.25636758\
 0.2473504  0.24577581 0.24507233 0.24484937 0.24399578 0.24310285\
 0.242081   0.24048809 0.24021001 0.23926089 0.23810557 0.23734479\
 0.23541522 0.23565347 0.23414027 0.23300291 0.23245488 0.23231026\
 0.23158682 0.23093207 0.23042265 0.22970915 0.22894992 0.22776375\
 0.22682536 0.22656425 0.22445137 0.22425304 0.22344221 0.22197653\
 0.22185734 0.22091239 0.21960432'
nopenbptt14to14 = nopenbptt14to14.split()
nopenbptt14to14 = list(map(float, nopenbptt14to14))
nopenbptt14to20 = '0.         0.12247453 0.3917361  0.5769519  0.68999992 0.77646837\
 0.8435238  0.90003064 0.94582194 0.91091181 0.77026601 0.59592623\
 0.46411174 0.3574494  0.28248666 0.23936505 0.21486582 0.20248709\
 0.19692795 0.19176011 0.19026805 0.18927077 0.18780496 0.18678488\
 0.1860507  0.18588839 0.18513698 0.1849109  0.1845514  0.18438268\
 0.18362818 0.18385367 0.18346514 0.1840288  0.18290671 0.18233495\
 0.18217358 0.18126486 0.18036067 0.18018402 0.17945465 0.17944701\
 0.178366   0.17857337 0.17749363 0.17812861 0.17758404 0.17810923\
 0.17757414 0.17740256 0.17606241'
nopenbptt14to20 = nopenbptt14to20.split()
nopenbptt14to20 = list(map(float, nopenbptt14to20))
nopenbptt14to25 = '0.         0.0876497  0.2957638  0.48334678 0.59674702 0.67862485\
 0.73809228 0.78076225 0.77084384 0.6584071  0.50598415 0.37832619\
 0.27609212 0.20888845 0.17564004 0.16376783 0.15943877 0.15771201\
 0.15740516 0.15620265 0.15514714 0.15469251 0.15407972 0.15269317\
 0.15117556 0.15087005 0.15020258 0.14960686 0.14863796 0.14758406\
 0.14660239 0.14656901 0.14454849 0.14394569 0.14286781 0.14201874\
 0.14114031 0.14062684 0.14016377 0.14030491 0.14016344 0.13984182\
 0.1395155  0.13910589 0.13832901 0.13771621 0.13738342 0.13690502\
 0.13612262 0.13626018 0.13547067'
nopenbptt14to25 = nopenbptt14to25.split()
nopenbptt14to25 = list(map(float, nopenbptt14to25))

nopentruncbptt14to14 = '0.         0.08885355 0.23669197 0.37825401 0.64734256 0.8439006\
 0.95633211 0.97368925 0.93359078 0.7302484  0.57020818 0.47311043\
 0.41071128 0.38765388 0.3739346  0.36423871 0.35288017 0.3498635\
 0.34729075 0.34262859 0.33250928 0.32515641 0.31412235 0.30346193\
 0.29435819 0.29100211 0.28557605 0.28203963 0.28064334 0.27590392\
 0.27132792 0.26703406 0.26347935 0.25869935 0.25606056 0.25498476\
 0.25638852 0.25773191 0.25745307 0.2596706  0.25901027 0.25736542\
 0.25461336 0.25396563 0.25220692 0.25180555 0.25314743 0.25416778\
 0.25449236 0.25460623 0.25357349'
nopentruncbptt14to14 = nopentruncbptt14to14.split()
nopentruncbptt14to14 = list(map(float, nopentruncbptt14to14))
nopentruncbptt14to20 = '0.         0.08885355 0.23669197 0.37825401 0.64734256 0.8439006\
 0.95633211 0.97368925 0.93359078 0.7302484  0.57020818 0.47311043\
 0.41071128 0.38765388 0.3739346  0.36423871 0.35288017 0.3498635\
 0.34729075 0.34262859 0.33250928 0.32515641 0.31412235 0.30346193\
 0.29435819 0.29100211 0.28557605 0.28203963 0.28064334 0.27590392\
 0.27132792 0.26703406 0.26347935 0.25869935 0.25606056 0.25498476\
 0.25638852 0.25773191 0.25745307 0.2596706  0.25901027 0.25736542\
 0.25461336 0.25396563 0.25220692 0.25180555 0.25314743 0.25416778\
 0.25449236 0.25460623 0.25357349'
nopentruncbptt14to20 = nopentruncbptt14to20.split()
nopentruncbptt14to20 = list(map(float, nopentruncbptt14to20))
nopentruncbptt14to25 = '0.         0.05332435 0.14532651 0.19854578 0.34338808 0.564368\
 0.68044243 0.75171939 0.70710095 0.52565217 0.36746137 0.28300908\
 0.24983483 0.24151108 0.24345159 0.24206301 0.24426577 0.24444272\
 0.24199752 0.235242   0.23137193 0.22621351 0.22156669 0.22167658\
 0.22228516 0.22202967 0.22148572 0.22159175 0.21964918 0.21654792\
 0.21229563 0.20813381 0.20463323 0.20253845 0.19989909 0.19988628\
 0.20130804 0.20005057 0.20226504 0.19927346 0.20104575 0.19692484\
 0.19850237 0.19523015 0.1969551  0.19475829 0.1976033  0.19756094\
 0.19956831 0.1976191  0.19920252'
nopentruncbptt14to25 = nopentruncbptt14to25.split()
nopentruncbptt14to25 = list(map(float, nopentruncbptt14to25))

wpenbptt14to14 = '0.         0.28086103 0.57224191 0.69573226 0.81539245 0.92215407\
 0.98747113 0.98263251 0.97651402 0.97286574 0.97028445 0.965962\
 0.96074763 0.95362341 0.94679989 0.9387376  0.93208553 0.92621633\
 0.92158895 0.91693884 0.91097018 0.90628635 0.90494402 0.90205537\
 0.9007064  0.89836853 0.89431871 0.88981344 0.88384192 0.88026199\
 0.87883771 0.87594514 0.87550835 0.87324407 0.87028555 0.86928268\
 0.86788134 0.86665225 0.8651589  0.86334057 0.85978896 0.85620508\
 0.85252398 0.85061801 0.84793446 0.84664482 0.8452114  0.84307442\
 0.84189134 0.83948023 0.83847869'
wpenbptt14to14 = wpenbptt14to14.split()
wpenbptt14to14 = list(map(float, wpenbptt14to14))
wpenbptt14to20 = '0.         0.217296   0.47832005 0.56072427 0.67396515 0.76704854\
 0.85346011 0.92424907 0.98517551 0.99012383 0.98376192 0.97210209\
 0.94949427 0.92141286 0.89293938 0.86973206 0.86282569 0.86493542\
 0.87142244 0.87416599 0.86968519 0.85985906 0.84671038 0.83838514\
 0.83199238 0.82704193 0.82599565 0.82823407 0.82661715 0.82534325\
 0.82389327 0.82134637 0.8196973  0.81649891 0.81560649 0.81417587\
 0.81287358 0.81110107 0.80942252 0.80626089 0.8042934  0.80188845\
 0.80036853 0.79827331 0.79721311 0.7951706  0.79332932 0.79261893\
 0.79206519 0.79101086 0.79038867'
wpenbptt14to20 = wpenbptt14to20.split()
wpenbptt14to20 = list(map(float, wpenbptt14to20))
wpenbptt14to25 = '0.         0.17742469 0.35045561 0.47378582 0.57855329 0.67189009\
 0.75469107 0.81816065 0.87031747 0.90527508 0.92461199 0.88836172\
 0.82137559 0.76680709 0.72997243 0.71922115 0.70973844 0.70925515\
 0.71301001 0.71648431 0.71925602 0.71357719 0.70523314 0.69748938\
 0.69270357 0.68907596 0.68498947 0.68278983 0.68190546 0.67926344\
 0.67689308 0.672937   0.67084841 0.6682775  0.66627674 0.66474145\
 0.66513011 0.66356932 0.6615145  0.66087937 0.65949903 0.65825251\
 0.65727541 0.6562009  0.65565659 0.65458509 0.65362996 0.65255985\
 0.65182318 0.65107136 0.65011918'
wpenbptt14to25 = wpenbptt14to25.split()
wpenbptt14to25 = list(map(float, wpenbptt14to25))

asbpttnopen14to14 = '0.         0.50841851 0.70562808 0.80887932 0.90008286 0.97355642\
 0.98516197 0.98405531 0.98144959 0.96748566 0.88151357 0.68226408\
 0.47551373 0.30982714 0.21305954 0.16801212 0.15251426 0.13997659\
 0.12926921 0.11532486 0.10858219 0.09672257 0.08364442 0.07121038\
 0.06032365 0.05234556 0.04769219 0.04378389 0.04053815 0.03843502\
 0.037246   0.03686588 0.03702091 0.03637034 0.03714452 0.03610414\
 0.03758868 0.03738478 0.03768112 0.03763984 0.03819921 0.03776133\
 0.0391032  0.03913713 0.03984632 0.03990389 0.03987713 0.03983417\
 0.04018172 0.04012897 0.03986245'
asbpttnopen14to14 = asbpttnopen14to14.split()
asbpttnopen14to14 = list(map(float, asbpttnopen14to14))

asbpttnopen14to20 = '0.         0.45946254 0.51432252 0.62821843 0.73160589 0.81689213\
 0.87417203 0.90811517 0.92272068 0.90016226 0.793032   0.59722386\
 0.39552276 0.25370325 0.18094103 0.140215   0.12522804 0.113271\
 0.10421026 0.09408428 0.08570469 0.08030305 0.07399416 0.06669114\
 0.06133553 0.05433065 0.04961658 0.0453166  0.04226121 0.03888236\
 0.03633486 0.03412032 0.03427307 0.03173296 0.03058056 0.02877109\
 0.02784622 0.02661028 0.0268223  0.02556761 0.02623564 0.02554291\
 0.02540809 0.02532761 0.02616221 0.02641538 0.02636728 0.02623433\
 0.02691483 0.02681625 0.02783678'
asbpttnopen14to20 = asbpttnopen14to20.split()
asbpttnopen14to20 = list(map(float, asbpttnopen14to20))

asbpttnopen14to25 = '0.         0.24879727 0.42951101 0.54766603 0.65645364 0.71773601\
 0.77285708 0.80372842 0.80417349 0.74683418 0.58148973 0.38514041\
 0.23769343 0.1577113  0.12291423 0.10231541 0.09412529 0.08835741\
 0.08291703 0.07812467 0.07514524 0.07024826 0.06567495 0.05987058\
 0.05385817 0.04821859 0.04517603 0.04220363 0.03970975 0.03859162\
 0.03688496 0.0364999  0.03615591 0.03650647 0.03659442 0.03737818\
 0.03702549 0.03784998 0.03854582 0.03852283 0.03835485 0.03876282\
 0.03884062 0.03926298 0.0394915  0.04021414 0.03990983 0.03968324\
 0.03998554 0.04003232 0.03959992'
asbpttnopen14to25 = asbpttnopen14to25.split()
asbpttnopen14to25 = list(map(float, asbpttnopen14to25))


lstmcbp14to14 = '0.         0.2359218  0.35391899 0.41296817 0.52718133 0.57680249\
 0.51008428 0.65023931 0.6566388  0.72392807 0.7314157  0.77095454\
 0.80751158 0.82922325 0.86923316 0.86555692 0.9142432  0.9012376\
 0.9424647  0.91776631 0.95099516 0.92633642 0.95704171 0.93510361\
 0.96108934 0.93975813 0.96243168 0.94272263 0.96299429 0.9446484\
 0.96406272 0.94493686 0.96366504 0.94718315 0.96356012 0.94770862\
 0.96475105 0.95027352 0.96407016 0.95108693 0.96430132 0.95137129\
 0.9643443  0.95261459 0.96489779 0.95269759 0.96414766 0.95296119\
 0.96352786 0.95509509 0.96474059 0.95540217 0.96380976 0.95620468\
 0.96356235 0.95731707 0.96492405 0.95709038 0.96352096 0.95719593\
 0.96582992 0.9573671  0.96460861 0.95732466 0.96425797 0.95747122\
 0.96409706 0.95675036 0.96409528 0.95703737 0.96447376'
lstmcbp14to14 = lstmcbp14to14.split()
lstmcbp14to14 = list(map(float, lstmcbp14to14))
lstmcbp14to20 = '0.         0.19589744 0.43666187 0.42021146 0.49556457 0.55212313\
 0.40581928 0.60543009 0.56506874 0.66522625 0.6408839  0.67372536\
 0.69916888 0.7161633  0.74895881 0.75493131 0.79064575 0.79587944\
 0.83322293 0.83088743 0.87429954 0.85870115 0.89998628 0.88297913\
 0.92024558 0.90112902 0.92910184 0.91093377 0.93791328 0.9188023\
 0.94294098 0.92547343 0.94389411 0.93059454 0.94862204 0.93303227\
 0.95013301 0.93681218 0.95157476 0.93841547 0.95294805 0.94176888\
 0.95451756 0.94226977 0.95505299 0.94317357 0.95456233 0.94582482\
 0.95669293 0.94335715 0.95381066 0.94485014 0.95616785 0.94548861\
 0.95553698 0.94623035 0.95415615 0.94613575 0.95541108 0.94695022\
 0.95420613 0.94629934 0.95505025 0.94711601 0.95499726 0.94676932\
 0.95321556 0.94588235 0.95461486 0.94853216 0.95251192'
lstmcbp14to20 = lstmcbp14to20.split()
lstmcbp14to20 = list(map(float, lstmcbp14to20))
lstmcbp14to25 = '0.         0.1705912  0.25705192 0.27907997 0.31457325 0.3994969\
 0.24556611 0.47095492 0.35786837 0.49346093 0.45328133 0.50520813\
 0.53414752 0.54100503 0.58904613 0.56634415 0.63085925 0.59498494\
 0.66661293 0.62368126 0.70291814 0.64915564 0.73586629 0.68061214\
 0.76326962 0.70490105 0.78825846 0.73525189 0.81155327 0.75695951\
 0.82908553 0.77327145 0.83778796 0.78452079 0.84979559 0.80122067\
 0.86168872 0.81110466 0.87003733 0.8204017  0.87284625 0.82578849\
 0.87811524 0.82792181 0.87941392 0.83246539 0.88230837 0.83523532\
 0.88170212 0.83759915 0.88280294 0.83533298 0.88215429 0.83818663\
 0.88436715 0.83650452 0.87926624 0.8393561  0.88212973 0.83780135\
 0.87761586 0.83873026 0.88202124 0.84190394 0.88331733 0.84496104\
 0.88484374 0.84666751 0.8861536  0.84864097 0.88698191'
lstmcbp14to25 = lstmcbp14to25.split()
lstmcbp14to25 = list(map(float, lstmcbp14to25))

lstmbpttnopen14to14 = '0.         0.32405338 0.46057607 0.6015393  0.72089004 0.81442637\
 0.91167333 0.93310287 0.94672589 0.9392288  0.932595   0.87155408\
 0.85211527 0.79710126 0.74771183 0.70201221 0.67030446 0.66021843\
 0.64205139 0.62673176 0.60741941 0.59941269 0.58082341 0.56606969\
 0.55643016 0.55281172 0.55085877 0.54667486 0.55568966 0.56457274\
 0.56175387 0.56580816 0.55913537 0.55332138 0.54834051 0.54325104\
 0.53856814 0.541373   0.54350964 0.54665527 0.53846102 0.54079624\
 0.52672985 0.52835558 0.51323048 0.51437512 0.50995654 0.51281184\
 0.51338296 0.51813282 0.5195802  0.53184466 0.53333084 0.5307142\
 0.53049919 0.53038773 0.52222257 0.5187741  0.51275487 0.51323631\
 0.51968897 0.51954155 0.51805775 0.51826937 0.51573362 0.51599857\
 0.51297662 0.51948693 0.51304293 0.52294896 0.51638376'
lstmbpttnopen14to14 = lstmbpttnopen14to14.split()
lstmbpttnopen14to14 = list(map(float, lstmbpttnopen14to14))
lstmbpttnopen14to20 = '0.         0.27343821 0.37691131 0.47556    0.57028029 0.64605202\
 0.71489199 0.77110699 0.82004875 0.83812235 0.85651904 0.80688656\
 0.80261948 0.73275382 0.67265529 0.62136718 0.58613337 0.55942258\
 0.53954969 0.53157712 0.51650945 0.51058268 0.4996954  0.48880346\
 0.48689317 0.47702299 0.47688056 0.47999759 0.47663094 0.46967823\
 0.47257355 0.47606249 0.47653691 0.47514535 0.48262656 0.48035056\
 0.49902983 0.4961408  0.49835926 0.48906787 0.49622287 0.48769661\
 0.48078162 0.47341092 0.47409675 0.46234651 0.46414153 0.45254674\
 0.46117447 0.45237331 0.44622373 0.4496538  0.45524675 0.45061854\
 0.45597784 0.45156528 0.44991322 0.44741137 0.45083853 0.4488123\
 0.45329109 0.44762832 0.44086673 0.44213455 0.44313739 0.44245052\
 0.43376781 0.44073561 0.4413977  0.43793735 0.43386427'
lstmbpttnopen14to20 = lstmbpttnopen14to20.split()
lstmbpttnopen14to20 = list(map(float, lstmbpttnopen14to20))
lstmbpttnopen14to25 = '0.         0.1913909  0.27617069 0.38685323 0.47252391 0.49738003\
 0.50938841 0.51376773 0.55546247 0.64761603 0.693231   0.68725022\
 0.69469553 0.64328776 0.58754918 0.53589284 0.49861595 0.46829938\
 0.44754448 0.42463229 0.41042476 0.40428568 0.3860916  0.36992649\
 0.35145831 0.33758521 0.33030353 0.32833088 0.3276479  0.33329634\
 0.32829989 0.33796018 0.34013919 0.34534796 0.34051439 0.33576969\
 0.3367921  0.33705314 0.33759822 0.33587394 0.3249995  0.32485068\
 0.31661434 0.31472062 0.30134108 0.30675062 0.30559726 0.30156044\
 0.29082071 0.2958525  0.29358878 0.29389822 0.28978446 0.28530407\
 0.28707487 0.28827566 0.28463853 0.2844453  0.28450129 0.28136154\
 0.27479912 0.26978973 0.26922514 0.25795688 0.26216219 0.261955\
 0.2662865  0.26270646 0.25963819 0.2631834  0.25671715'
lstmbpttnopen14to25 = lstmbpttnopen14to25.split()
lstmbpttnopen14to25 = list(map(float, lstmbpttnopen14to25))

# Also add mix-data here
bpttnopen_14tomix = '0.         0.14722878 0.44664368 0.64589477 0.7665645  0.85107926\
 0.90201871 0.93033688 0.9274787  0.86417393 0.73201246 0.57905929\
 0.44989667 0.35251728 0.29364246 0.25747325 0.24075229 0.23388681\
 0.23089618 0.22878597 0.22608942 0.22545942 0.2241642  0.22334583\
 0.22315139 0.22222595 0.22136536 0.22074665 0.21988316 0.21951678\
 0.21968172 0.21825138 0.21749764 0.21718154 0.21661448 0.21604595\
 0.21575253 0.21502961 0.21454673 0.21412027 0.21383641 0.21337234\
 0.21311962 0.21233969 0.21128699 0.21125648 0.2103411  0.20974641\
 0.20852292 0.20836234 0.20751457'
bpttnopen_14tomix = bpttnopen_14tomix.split()
bpttnopen_14tomix = list(map(float, bpttnopen_14tomix))
bpttpen_14tomix = '0.         0.26159805 0.51309474 0.63603968 0.75436165 0.84597976\
 0.90617997 0.94646627 0.97012798 0.97779819 0.97257743 0.95743183\
 0.93893585 0.92179458 0.9083861  0.8971682  0.89317289 0.89115302\
 0.88742077 0.88506951 0.88176767 0.8778188  0.87412023 0.87139159\
 0.86925502 0.86670362 0.86542343 0.86219228 0.86006418 0.85661865\
 0.85115312 0.84797361 0.84454857 0.84155066 0.83868666 0.83574348\
 0.83315083 0.82985368 0.82698117 0.8249053  0.82402419 0.82258719\
 0.82134782 0.82017434 0.81975192 0.8182339  0.81545036 0.81235359\
 0.81022014 0.80855892 0.80634891'
bpttpen_14tomix = bpttpen_14tomix.split()
bpttpen_14tomix = list(map(float, bpttpen_14tomix))
rbppen_14tomix = '0.         0.1671529  0.24428714 0.26460654 0.334475   0.36661771\
 0.49473414 0.56127807 0.64379246 0.70008199 0.75555605 0.79507913\
 0.83365126 0.85976094 0.88495214 0.90435826 0.92097982 0.93404792\
 0.94503678 0.9567325  0.96078787 0.96531083 0.96778075 0.96911654\
 0.97093395 0.97169076 0.97097308 0.97093166 0.96947855 0.9693828\
 0.9685974  0.96845779 0.96806553 0.96783104 0.96625859 0.96681539\
 0.96673137 0.96672841 0.96702706 0.96692964 0.96663139 0.96628307\
 0.96580159 0.96561506 0.96610269 0.96538232 0.96563523 0.96504527\
 0.96504671 0.96462374 0.96432762'
rbppen_14tomix = rbppen_14tomix.split()
rbppen_14tomix = list(map(float, rbppen_14tomix))

# Start plotting
df = pd.DataFrame(
    np.hstack((
        np.stack((
            cbp14to14[:40],
            cbp14to20[:40],
            cbp14to25[:40],
            nopenbptt14to14[:40],
            nopenbptt14to20[:40],
            nopenbptt14to25[:40],
            wpenbptt14to14[:40],
            wpenbptt14to20[:40],
            wpenbptt14to25[:40],
            ), 0
        ).reshape(-1, 1),
        np.arange(40).reshape(-1, 1).repeat(9, 1).transpose().reshape(-1, 1),  # noqa
        np.stack((
            'CBP',
            'CBP',
            'CBP',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/ penalty',
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            colors['cbp'],
            colors['cbp'],
            colors['cbp'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        )).reshape(1, -1).repeat(40, 1).transpose(),
    )),
    columns=['IOU', 'Steps', 'Model', 'Dataset', 'color', 'style'])
df.IOU = f_to_iou(pd.to_numeric(df.IOU))
df.Steps = pd.to_numeric(df.Steps)
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df[df.Dataset == '14'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[0])
# box = g.get_position()
axs[0].set_ylim([-0.05, 1.05])
axs[0].set_title('Trained on 14, testing on 14.')
g = sns.lineplot(
    data=df[df.Dataset == '20'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[1])
# box = g.get_position()
# g.set_position([box.x0, box.y0, box.width * 0.95, box.height])
# g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
axs[1].set_ylim([-0.05, 1.05])
axs[1].set_title('Trained on 14, testing on 20.')
g = sns.lineplot(
    data=df[df.Dataset == '25'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[2])
# box = g.get_position()
# g.set_position([box.x0, box.y0, box.width * 0.95, box.height])
# g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
axs[2].set_ylim([-0.05, 1.05])
axs[2].set_title('Trained on 14, testing on 20.')
plt.show()
plt.close(fig)

# Slopes
scores = df.groupby(['Model', 'Dataset']).max().reset_index()
models = scores.Model.unique()
num_datasets = len(df.Dataset.unique())
slopes = []
for m in models:
    slope = np.corrcoef(
        np.arange(num_datasets),
        scores[scores.Model == m].IOU)[0, 1]
    slopes.append(slope)
    print(m, slope)

# Control
df = pd.DataFrame(
    np.hstack((
        np.stack((
            asbpttnopen14to14[:40],
            asbpttnopen14to20[:40],
            asbpttnopen14to25[:40],
            # wpenbptt14to14[:40],
            # wpenbptt14to20[:40],
            # wpenbptt14to25[:40
            cbp14to14[:40],
            cbp14to20[:40],
            cbp14to25[:40],
            nopenbptt14to14[:40],
            nopenbptt14to20[:40],
            nopenbptt14to25[:40],
            nopentruncbptt14to14[:40],
            nopentruncbptt14to20[:40],
            nopentruncbptt14to25[:40],
            ), 0
        ).reshape(-1, 1),
        np.arange(40).reshape(-1, 1).repeat(12 , 1).transpose().reshape(-1, 1),  # noqa
        np.stack((
            'BPTT w/o penalty and timestep supervision',
            'BPTT w/o penalty and timestep supervision',
            'BPTT w/o penalty and timestep supervision',
            # 'BPTT w/ penalty',
            # 'BPTT w/ penalty',
            # 'BPTT w/ penalty',
            'CBP',
            'CBP',
            'CBP',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty truncated',
            'BPTT w/o penalty truncated',
            'BPTT w/o penalty truncated',
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            # '14',
            # '20',
            # '25',
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            colors['bptt_wo_penalty_allsup'],
            colors['bptt_wo_penalty_allsup'],
            colors['bptt_wo_penalty_allsup'],
            # colors['bptt_w_penalty'],
            # colors['bptt_w_penalty'],
            # colors['bptt_w_penalty'],
            colors['cbp'],
            colors['cbp'],
            colors['cbp'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty_truncated'],
            colors['bptt_wo_penalty_truncated'],
            colors['bptt_wo_penalty_truncated'],
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        )).reshape(1, -1).repeat(40, 1).transpose(),
    )),
    columns=['IOU', 'Steps', 'Model', 'Dataset', 'color', 'style'])
df.IOU = f_to_iou(pd.to_numeric(df.IOU))
df.Steps = pd.to_numeric(df.Steps)
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df[df.Dataset == '14'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[0])
# box = g.get_position()
axs[0].set_ylim([-0.05, 1.05])
axs[0].set_title('Trained on 14, testing on 14.')
g = sns.lineplot(
    data=df[df.Dataset == '20'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[1])
# box = g.get_position()
# g.set_position([box.x0, box.y0, box.width * 0.95, box.height])
# g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
axs[1].set_ylim([-0.05, 1.05])
axs[1].set_title('Trained on 14, testing on 20.')
g = sns.lineplot(
    data=df[df.Dataset == '25'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[2])
# box = g.get_position()
# g.set_position([box.x0, box.y0, box.width * 0.95, box.height])
# g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
axs[2].set_ylim([-0.05, 1.05])
axs[2].set_title('Trained on 14, testing on 20.')
plt.show()
plt.close(fig)

# Slopes
scores = df.groupby(['Model', 'Dataset']).max().reset_index()
models = scores.Model.unique()
num_datasets = len(df.Dataset.unique())
slopes = []
for m in models:
    slope = np.corrcoef(
        np.arange(num_datasets),
        scores[scores.Model == m].IOU)[0, 1]
    slopes.append(slope)
    print(m, slope)

# Control
df = pd.DataFrame(
    np.hstack((
        np.stack((
            cbp14to14[:40],
            cbp14to20[:40],
            cbp14to25[:40],
            lstmcbp14to14[:40],
            lstmcbp14to20[:40],
            lstmcbp14to25[:40],
            lstmbpttnopen14to14[:40],
            lstmbpttnopen14to20[:40],
            lstmbpttnopen14to25[:40],
            ), 0
        ).reshape(-1, 1),
        np.arange(40).reshape(-1, 1).repeat(9 , 1).transpose().reshape(-1, 1),  # noqa
        np.stack((
            'CBP',
            'CBP',
            'CBP',
            'LSTM CBP',
            'LSTM CBP',
            'LSTM CBP',
            'LSTM BPTT w/o penalty',
            'LSTM BPTT w/o penalty',
            'LSTM BPTT w/o penalty',
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            colors['cbp'],
            colors['cbp'],
            colors['cbp'],
            colors['lstm_cbp'],
            colors['lstm_cbp'],
            colors['lstm_cbp'],
            colors['lstm_bptt_wo_penalty'],
            colors['lstm_bptt_wo_penalty'],
            colors['lstm_bptt_wo_penalty'],
        )).reshape(1, -1).repeat(40, 1).transpose(),
        np.stack((
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        )).reshape(1, -1).repeat(40, 1).transpose(),
    )),
    columns=['IOU', 'Steps', 'Model', 'Dataset', 'color', 'style'])
df.IOU = f_to_iou(pd.to_numeric(df.IOU))
df.Steps = pd.to_numeric(df.Steps)
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df[df.Dataset == '14'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[0])
# box = g.get_position()
axs[0].set_ylim([-0.05, 1.05])
axs[0].set_title('Trained on 14, testing on 14.')
g = sns.lineplot(
    data=df[df.Dataset == '20'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[1])
# box = g.get_position()
# g.set_position([box.x0, box.y0, box.width * 0.95, box.height])
# g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
axs[1].set_ylim([-0.05, 1.05])
axs[1].set_title('Trained on 14, testing on 20.')
g = sns.lineplot(
    data=df[df.Dataset == '25'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    legend=False,
    ax=axs[2])
# box = g.get_position()
# g.set_position([box.x0, box.y0, box.width * 0.95, box.height])
# g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
axs[2].set_ylim([-0.05, 1.05])
axs[2].set_title('Trained on 14, testing on 20.')
plt.show()
plt.close(fig)

# Slopes
scores = df.groupby(['Model', 'Dataset']).max().reset_index()
models = scores.Model.unique()
num_datasets = len(df.Dataset.unique())
slopes = []
for m in models:
    slope = np.corrcoef(
        np.arange(num_datasets),
        scores[scores.Model == m].IOU)[0, 1]
    slopes.append(slope)
    print(m, slope)

# # Example images for train/test same dataset
# 14/14
cmap = 'Greys_r'
input_im_14 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/0_inp.png'  # noqa
input_lab_14 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/0_inp.png'  # noqa
bptt_nopen_14_1 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttnopen14/14/0_1_op.png'  # noqa
bptt_nopen_14_2 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttnopen14/14/0_6_op.png'  # noqa
bptt_nopen_14_3 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttnopen14/14/0_40_op.png'  # noqa
bptt_pen_14_1 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/0_1_op.png'  # noqa
bptt_pen_14_2 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/0_6_op.png'  # noqa
bptt_pen_14_3 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/0_40_op.png'  # noqa
rbp_pen_14_1 = '/Users/drewlinsley/Downloads/rbp_results/figs/rbppen14/14/0_1_op.png'  # noqa
rbp_pen_14_2 = '/Users/drewlinsley/Downloads/rbp_results/figs/rbppen14/14/0_22_op.png'  # noqa
rbp_pen_14_3 = '/Users/drewlinsley/Downloads/rbp_results/figs/rbppen14/14/0_40_op.png'  # noqa
fig = plt.figure()
plt.subplot(4, 3, 1)
plt.axis('off')
plt.imshow(io.imread(input_im_14), cmap=cmap)
plt.subplot(4, 3, 3)
plt.axis('off')
plt.imshow(io.imread(input_lab_14), cmap=cmap)
plt.subplot(4, 3, 4)
plt.axis('off')
plt.imshow(np.flipud(io.imread(bptt_nopen_14_1)), cmap=cmap)
plt.subplot(4, 3, 5)
plt.axis('off')
plt.imshow(io.imread(bptt_pen_14_1), cmap=cmap)
plt.subplot(4, 3, 6)
plt.axis('off')
plt.imshow(np.rot90(io.imread(rbp_pen_14_1), -2), cmap=cmap)
plt.subplot(4, 3, 7)
plt.axis('off')
plt.imshow(np.flipud(io.imread(bptt_nopen_14_2)), cmap=cmap)
plt.subplot(4, 3, 8)
plt.axis('off')
plt.imshow(io.imread(bptt_pen_14_2), cmap=cmap)
plt.subplot(4, 3, 9)
plt.axis('off')
plt.imshow(np.rot90(io.imread(rbp_pen_14_2), -2), cmap=cmap)
plt.subplot(4, 3, 10)
plt.axis('off')
plt.imshow(np.flipud(io.imread(bptt_nopen_14_3)), cmap=cmap)
plt.subplot(4, 3, 11)
plt.axis('off')
plt.imshow(io.imread(bptt_pen_14_3), cmap=cmap)
plt.subplot(4, 3, 12)
plt.axis('off')
plt.imshow(np.rot90(io.imread(rbp_pen_14_3), -2), cmap=cmap)
plt.show()

# Extrapolation to 14/20/25
cmap = 'Greys_r'
input_im_14 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/1_inp.png'  # noqa
input_lab_14 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/1_lab.png'  # noqa
input_im_20 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/20/22_inp.png'  # noqa
input_lab_20 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/20/22_lab.png'  # noqa
input_im_25 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/25/5_inp.png'  # noqa
input_lab_25 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/25/5_lab.png'  # noqa

fig = plt.figure()
plt.subplot(2, 3, 1)
plt.axis('off')
plt.imshow(io.imread(input_im_14), cmap=cmap)
plt.subplot(2, 3, 2)
plt.axis('off')
plt.imshow(io.imread(input_im_20), cmap=cmap)
plt.subplot(2, 3, 3)
plt.axis('off')
plt.imshow(io.imread(input_im_25), cmap=cmap)
plt.subplot(2, 3, 4)
plt.axis('off')
plt.imshow(io.imread(input_lab_14), cmap=cmap)
plt.subplot(2, 3, 5)
plt.axis('off')
plt.imshow(io.imread(input_lab_20), cmap=cmap)
plt.subplot(2, 3, 6)
plt.axis('off')
plt.imshow(io.imread(input_lab_25), cmap=cmap)
plt.show()
plt.close(fig)

bptt_nopen_14 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttnopen14/14/1_6_op.png'  # noqa
bptt_nopen_20 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttnopen14/20/22_8_op.png'  # noqa
bptt_nopen_25 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttnopen14/25/5_7_op.png'  # noqa
bptt_pen_14 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/14/1_6_op.png'  # noqa
bptt_pen_20 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/20/22_9_op.png'  # noqa
bptt_pen_25 = '/Users/drewlinsley/Downloads/rbp_results/figs/bpttpen14/25/5_10_op.png'  # noqa
rbp_pen_14 = '/Users/drewlinsley/Downloads/rbp_results/figs/rbppen14/14/1_22_op.png'  # noqa
rbp_pen_20 = '/Users/drewlinsley/Downloads/rbp_results/figs/rbppen14/20/22_25_op.png'  # noqa
rbp_pen_25 = '/Users/drewlinsley/Downloads/rbp_results/figs/rbppen14/25/5_37_op.png'  # noqa

plt.subplot(3, 3, 1)
plt.axis('off')
plt.imshow((io.imread(bptt_nopen_14)), cmap=cmap)
plt.subplot(3, 3, 2)
plt.axis('off')
plt.imshow(io.imread(bptt_pen_14), cmap=cmap)
plt.subplot(3, 3, 3)
plt.axis('off')
plt.imshow(np.rot90(io.imread(rbp_pen_14), 0), cmap=cmap)
plt.subplot(3, 3, 4)
plt.axis('off')
plt.imshow((io.imread(bptt_nopen_20)), cmap=cmap)
plt.subplot(3, 3, 5)
plt.axis('off')
plt.imshow(io.imread(bptt_pen_20), cmap=cmap)
plt.subplot(3, 3, 6)
plt.axis('off')
plt.imshow(np.rot90(io.imread(rbp_pen_20), 0), cmap=cmap)
plt.subplot(3, 3, 7)
plt.axis('off')
plt.imshow((io.imread(bptt_nopen_25)), cmap=cmap)
plt.subplot(3, 3, 8)
plt.axis('off')
plt.imshow(io.imread(bptt_pen_25), cmap=cmap)
plt.subplot(3, 3, 9)
plt.axis('off')
plt.imshow(np.rot90(io.imread(rbp_pen_25), 0), cmap=cmap)
plt.show()

# Training performance
pf14_bptt_pen = np.load(os.path.join('BPTT_penalty_hgru_PF14SEG_6ts_4GPU_128Batch', 'train.npz'))  # noqa
pf20_bptt_pen = np.load(os.path.join('BPTT_penalty_hgru_PF20SEG_6ts_4GPU_128Batch', 'train.npz'))  # noqa
pf25_bptt_pen = np.load(os.path.join('BPTT_penalty_hgru_PF25SEG_6ts_4GPU_128Batch', 'train.npz'))  # noqa
pf14_bptt_nopen = np.load(os.path.join('BPTT_nopen_hgru_PF14SEG_6ts_4GPU_128Batch_P4', 'train.npz'))  # noqa
pf20_bptt_nopen = np.load(os.path.join('BPTT_nopen_hgru_PF20SEG_6ts_4GPU_128Batch_P4', 'train.npz'))  # noqa
pf25_bptt_nopen = np.load(os.path.join('BPTT_nopen_hgru_PF25SEG_6ts_4GPU_128Batch', 'train.npz'))  # noqa
pf14_cbp = np.load(os.path.join('RBP_penalty_hgru_PF14SEG_20ts_15iter_4GPU_128Batch_P4', 'train.npz'))  # noqa
pf20_cbp = np.load(os.path.join('RBP_penalty_hgru_PF20SEG_30ts_15iter_4GPU_128Batch', 'train.npz'))  # noqa
pf25_cbp = np.load(os.path.join('RBP_penalty_hgru_PF25SEG_80ts_15iter_4GPU_128Batch', 'train.npz'))  # noqa
pf14_rbp = np.load(os.path.join('RBP_nopen_hgru_PF14SEG_20ts_20iter_4GPU_128Batch_P5', 'train.npz'))  # noqa
pf20_rbp = np.load(os.path.join('RBP_nopen_hgru_PF20SEG_20ts_15iter_4GPU_128Batch_P5', 'train.npz'))  # noqa
pf25_rbp = np.load(os.path.join('RBP_nopen_hgru_PF25SEG_50ts_15iter_4GPU_128Batch', 'train.npz'))  # noqa

pf14_bptt_pen_train = pf14_bptt_pen['f1score']
pf20_bptt_pen_train = pf20_bptt_pen['f1score']
pf25_bptt_pen_train = pf25_bptt_pen['f1score']
pf14_bptt_nopen_train = pf14_bptt_nopen['f1score']
pf20_bptt_nopen_train = pf20_bptt_nopen['f1score']
pf25_bptt_nopen_train = pf25_bptt_nopen['f1score']
pf14_cbp_train = pf14_cbp['f1score']
pf20_cbp_train = pf20_cbp['f1score']
pf25_cbp_train = pf25_cbp['f1score']
pf14_rbp_train = pf14_rbp['f1score']
pf20_rbp_train = pf20_rbp['f1score']
pf25_rbp_train = pf25_rbp['f1score']

df = pd.DataFrame(
    np.hstack((
        np.stack((
            pf14_bptt_pen_train[:25848],
            pf20_bptt_pen_train[:25848],
            pf25_bptt_pen_train[:25848],
            pf14_bptt_nopen_train[:25848],
            pf20_bptt_nopen_train[:25848],
            pf25_bptt_nopen_train[:25848],
            pf14_cbp_train[:25848],
            pf20_cbp_train[:25848],
            pf25_cbp_train[:25848],
            pf14_rbp_train[:25848],
            pf20_rbp_train[:25848],
            pf25_rbp_train[:25848],
            ), 0).reshape(-1, 1),
        np.arange(25848).reshape(-1, 1).repeat(12, 1).transpose().reshape(-1, 1),  # noqa
        np.stack((
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/ penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'BPTT w/o penalty',
            'CBP',
            'CBP',
            'CBP',
            'RBP',
            'RBP',
            'RBP',
        )).reshape(1, -1).repeat(25848, 1).transpose(),
        np.stack((
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
            '14',
            '20',
            '25',
        )).reshape(1, -1).repeat(25848, 1).transpose(),
        np.stack((
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_w_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['cbp'],
            colors['cbp'],
            colors['cbp'],
            colors['rbp'],
            colors['rbp'],
            colors['rbp'],
        )).reshape(1, -1).repeat(25848, 1).transpose(),
        np.stack((
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        )).reshape(1, -1).repeat(25848, 1).transpose(),
    )),
    columns=['IOU', 'Steps', 'Model', 'Dataset', 'color', 'style'])
df.IOU = f_to_iou(pd.to_numeric(df.IOU))
df.Steps = pd.to_numeric(df.Steps)
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df[df.Dataset == '14'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    lw=0.1,
    legend=False,
    ax=axs[0])
box = g.get_position()
axs[0].set_title('Training on 14.')
axs[0].set_ylim([-0.05, 1.05])
g = sns.lineplot(
    data=df[df.Dataset == '20'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    lw=0.1,
    legend=False,
    ax=axs[1])
box = g.get_position()
axs[1].set_title('Training on 20.')
axs[0].set_ylim([-0.05, 1.05])
g = sns.lineplot(
    data=df[df.Dataset == '25'],
    y='IOU',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    lw=0.1,
    legend=False,
    ax=axs[2])
box = g.get_position()
axs[2].set_title('Training on 25.')
axs[0].set_ylim([-0.05, 1.05])
plt.show()

# PF on different kernel sizes
pf14_bptt_nopen_3 = np.load(os.path.join('BPTT_nopen_hgru_3filt_PF14SEG_6ts_4GPU_128Batch', 'val.npz'))  # noqa
pf14_bptt_nopen_5 = np.load(os.path.join('BPTT_nopen_hgru_5filt_PF14SEG_6ts_4GPU_128Batch', 'val.npz'))  # noqa
pf14_bptt_nopen_9 = np.load(os.path.join('BPTT_nopen_hgru_9filt_PF14SEG_6ts_4GPU_128Batch_P4', 'val.npz'))  # noqa
pf14_cbp_3 = np.load(os.path.join('RBP_penalty_hgru_3filt_PF14SEG_300ts_15iter_8GPU_128Batch', 'val.npz'))  # noqa
pf14_cbp_5 = np.load(os.path.join('RBP_penalty_hgru_5filt_PF14SEG_100ts_15iter_8GPU_128Batch', 'val.npz'))  # noqa
pf14_cbp_9 = np.load(os.path.join('RBP_penalty_hgru_9filt_PF14SEG_30ts_15iter_4GPU_128Batch', 'val.npz'))  # noqa
df = pd.DataFrame(
    np.vstack((
        np.stack((
            pf14_bptt_nopen_3['f1score'].max(),
            pf14_bptt_nopen_5['f1score'].max(),
            pf14_bptt_nopen_9['f1score'].max(),
            pf14_cbp_3['f1score'].max(),
            pf14_cbp_5['f1score'].max(),
            pf14_cbp_9['f1score'].max(),
        )),
        np.stack((
            'BPTT',
            'BPTT',
            'BPTT',
            'CBP',
            'CBP',
            'CBP',
        )),
        np.stack((
            '3',
            '5',
            '9',
            '3',
            '5',
            '9',
        )),
        np.stack((
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['bptt_wo_penalty'],
            colors['cbp'],
            colors['cbp'],
            colors['cbp'],
        )),
        np.stack((
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        )),
    )).transpose(),
    columns=['IOU', 'Model', 'Kernel', 'color', 'style'])
df.IOU = f_to_iou(pd.to_numeric(df.IOU))
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df,
    y='IOU',
    x='Kernel',
    hue='Model',
    palette=palette,
    style='style',
    marker='h',
    lw=1,
    ax=ax1)
plt.ylim([0.49, 1.01])
box = g.get_position()
g.set_position([box.x0, box.y0, box.width * 0.65, box.height])
g.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), ncol=1)
plt.title('Training/testing on the same PF dataset.')
plt.show()
plt.close(fig)

# Vector jacobian product
filepath = os.path.join(
    'BPTT_nopen_hgru_PF14SEG_6ts_4GPU_128Batch_P4',
    'BPTT_nopen_hgru_PF14SEG_6ts_4GPU_128Batch_P4.txt')
lines = []
with open(filepath) as fp:
    for line in fp:
        if 'jvpen' in line:
            lines.append(float(line.split('jvpen: ')[-1].split(' ')[0]))
bptt_nopen_pf14 = np.array(lines)
filepath = os.path.join(
    'BPTT_penalty_hgru_PF14SEG_6ts_4GPU_128Batch',
    'BPTT_penalty_hgru_PF14SEG_6ts_4GPU_128Batch.txt')
lines = []
with open(filepath) as fp:
    for line in fp:
        if 'jvpen' in line:
            lines.append(float(line.split('jvpen: ')[-1].split(' ')[0]))
bptt_pen_pf14 = np.array(lines)
filepath = os.path.join(
    'RBP_nopen_hgru_PF14SEG_20ts_20iter_4GPU_128Batch_P5',
    'RBP_nopen_hgru_PF14SEG_20ts_20iter_4GPU_128Batch_P5.txt')
lines = []
with open(filepath) as fp:
    for line in fp:
        if 'jvpen' in line:
            lines.append(float(line.split('jvpen: ')[-1].split(' ')[0]))
rbp_pf14 = np.array(lines)
filepath = os.path.join(
    'RBP_penalty_hgru_PF14SEG_20ts_15iter_4GPU_128Batch_P4',
    'RBP_penalty_hgru_PF14SEG_20ts_15iter_4GPU_128Batch_P4.txt')
lines = []
with open(filepath) as fp:
    for line in fp:
        if 'jvpen' in line:
            lines.append(float(line.split('jvpen: ')[-1].split(' ')[0]))
cbp_pf14 = np.array(lines)
df = pd.DataFrame(
    np.hstack((
        np.stack((
            bptt_nopen_pf14,
            bptt_pen_pf14,
            rbp_pf14,
            cbp_pf14), 0).reshape(-1, 1),
        np.arange(1500).reshape(-1, 1).repeat(4, 1).transpose().reshape(-1, 1),  # noqa
        np.stack((
            'BPTT w/o penalty',
            'BPTT w/ penalty',
            'RBP',
            'CBP',
        )).reshape(1, -1).repeat(1500, 1).transpose(),
        np.stack((
            '14',
            '14',
            '14',
            '14',
        )).reshape(1, -1).repeat(1500, 1).transpose(),
        np.stack((
            colors['bptt_wo_penalty'],
            colors['bptt_w_penalty'],
            colors['rbp'],
            colors['cbp'],
        )).reshape(1, -1).repeat(1500, 1).transpose(),
        np.stack((
            '-',
            '-',
            '-',
            '-',
        )).reshape(1, -1).repeat(1500, 1).transpose(),
    )),
    columns=['VJ', 'Steps', 'Model', 'Dataset', 'color', 'style'])
df.VJ = np.log2(pd.to_numeric(df.VJ))
df.Steps = pd.to_numeric(df.Steps)
palette = {k: v for k, v in zip(df.Model, df.color)}
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.lineplot(
    data=df,
    y='VJ',
    x='Steps',
    hue='Model',
    palette=palette,
    style='style',
    lw=1.,
    legend=False,
    ax=axs)
box = g.get_position()
axs.set_title('Trained on 14')
plt.show()

# Plot weight distributions for input output gates
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
bptt_checkpoint = torch.load(
    os.path.join(
        'BPTT_nopen_hgru_PF14SEG_6ts_4GPU_128Batch_P4',
        'saved_models',
        'model_fscore_9931_epoch_19_checkpoint.pth.tar'), map_location=torch.device('cpu'))  # noqa
rbp_checkpoint = torch.load(
    os.path.join(
        'RBP_penalty_hgru_PF14SEG_20ts_15iter_4GPU_128Batch_P4',
        'saved_models',
        'model_fscore_9804_epoch_14_checkpoint.pth.tar'), map_location=torch.device('cpu'))  # noqa
# convs = rbp_checkpoint['state_dict']['module.conv0.weight']
bptt_output = bptt_checkpoint['state_dict']['module.unit1.w_gate_exc'].squeeze()
bptt_input = bptt_checkpoint['state_dict']['module.unit1.w_gate_inh'].squeeze()
rbp_output = rbp_checkpoint['state_dict']['module.unit1.w_gate_exc'].squeeze()
rbp_input = rbp_checkpoint['state_dict']['module.unit1.w_gate_inh'].squeeze()
sns.distplot(bptt_output.squeeze(), color='blue', label='bptt_excitation', ax=axs[0])  # noqa
sns.distplot(bptt_input.squeeze(), color='green', label='bptt_inhibition', ax=axs[0])  # noqa
sns.distplot(rbp_output.squeeze(), color='red', label='cbp_excitation', ax=axs[0])
sns.distplot(rbp_input.squeeze(), color='purple', label='cbp_inhibition', ax=axs[0])
axs[0].set_ylabel('Frequency')
axs[0].set_title('PF14')
axs[0].set_ylim(0, 8)
