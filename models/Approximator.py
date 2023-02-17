import pandas as pd
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import scipy.stats as sts
from scipy import signal

from tools.WLambert_Transform import *
from tools.data_preprocessing import *
from tools.operators import *
from tools.time_series_clustering import *

import models.TCBGAN as gan
import models.Clustering_GAN as ssagan
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose

def smoothing(x, alpha=0.35):
    ses = SimpleExpSmoothing(x, initialization_method="heuristic").fit(
    smoothing_level=alpha, optimized=False)

    return ses.fittedvalues

def regime_smoothing(synth, real, idx_cp, splitted_real, beta=0.005, win_size=300, smooth_trend=False):
    switch_regimes = copy.deepcopy(idx_cp)
    array = []
    for i in range(len(switch_regimes) - 1):
            array.append(synth[switch_regimes[i]:switch_regimes[i+1]])

    for i in range(len(array) - 1):
        delta = array[i][-1] - array[i+1][0]
        delta_r = splitted_real[i][-1] - splitted_real[i+1][0]

        if (delta >= 0) and (np.sign(delta) == np.sign(delta_r)):

            if abs(delta) >= abs(delta_r):
                array[i+1] = array[i+1] + (abs(delta) - abs(delta_r))

            elif abs(delta) < abs(delta_r):
                array[i+1] = array[i+1] - (abs(delta_r) - abs(delta))

            else:
                continue

        elif (delta < 0) and (np.sign(delta) == np.sign(delta_r)):

            if abs(delta) >= abs(delta_r):
                array[i+1] = array[i+1] - (abs(delta) - abs(delta_r))

            elif abs(delta) < abs(delta_r):
                array[i+1] = array[i+1] + (abs(delta_r) - abs(delta))

            else:
                continue

        elif (delta >= 0) and (np.sign(delta) != np.sign(delta_r)):
            array[i+1] = array[i+1] + (abs(delta) + abs(delta_r))

        elif (delta < 0) and (np.sign(delta) != np.sign(delta_r)):
            array[i+1] = array[i+1] - (abs(delta) + abs(delta_r))

        else:
            continue

#     for i in range(len(array) - 1):
#         # p = np.concatenate([splitted_real[i][-n_ch:], splitted_real[i+1][:n_ch]])
#         # p_tilde = np.concatenate([regimes_tilde_copy[i][-n_ch:], regimes_tilde_copy[i+1][:n_ch]])
#         p = np.concatenate([splitted_real[i][-int(len(regimes_tilde_copy[i])/n_ch):], splitted_real[i+1][:int(len(regimes_tilde_copy[i+1])/n_ch)]])
#         p_tilde = np.concatenate([regimes_tilde_copy[i][-int(len(regimes_tilde_copy[i])/n_ch):], regimes_tilde_copy[i+1][:int(len(regimes_tilde_copy[i+1])/n_ch)]])


#         # sig = p_tilde
#         # win = signal.windows.hann(3)
#         # p_smoothed = signal.convolve(sig, win, mode='same') / sum(win)
#         p_smoothed = 0.5 * (p + p_tilde)
# #         p_smoothed = (0.2*p_smoothed + 0.8*p_smoothed2)/2
#         # p_smoothed = smoothing(p_smoothed, alpha=0.5)

#         # regimes_tilde_copy[i][-n_ch:] = p_smoothed[:n_ch]
#         # regimes_tilde_copy[i+1][:n_ch] = p_smoothed[n_ch:]
#         regimes_tilde_copy[i][-int(len(regimes_tilde_copy[i])/n_ch):] = p_smoothed[:int(len(regimes_tilde_copy[i])/n_ch)]
#         regimes_tilde_copy[i+1][:int(len(regimes_tilde_copy[i+1])/n_ch)] = p_smoothed[int(len(regimes_tilde_copy[i])/n_ch):]

#     result = np.hstack(regimes_tilde_copy)

    # df_decomp_r = pd.DataFrame(real)
    # df_decomp_r.index = pd.date_range(start='1/1/2022', periods=len(real), freq='D')
    #
    # decomposed_real = seasonal_decompose(df_decomp_r, model='additive', extrapolate_trend='freq')
    # residuals_r = decomposed_real.resid.dropna().values
    # seasonal_r = decomposed_real.seasonal.dropna().values
    # trend_r = decomposed_real.trend.dropna().values


#     df_decomp = pd.DataFrame(synth)
#     df_decomp.index = pd.date_range(start='1/1/2022', periods=len(synth), freq='D')

#     decomposed = seasonal_decompose(df_decomp, model='additive', extrapolate_trend='freq')
#     residuals = decomposed.resid.dropna().values
#     seasonal = decomposed.seasonal.dropna().values
#     trend = decomposed.trend.dropna().values

    # result=synth
    #
    # switch_regimes = copy.deepcopy(idx_cp)
    # array = []
    # for i in range(len(switch_regimes) - 1):
    #         array.append(result[switch_regimes[i]:switch_regimes[i+1]])
    #
    # for i in range(len(array) - 1):
    #     delta = array[i][-1] - array[i+1][0]
    #
    #     if delta >= 0:
    #         array[i+1] = array[i+1] + delta
    #
    #     else:
    #         array[i+1] = array[i+1] - abs(delta)

    result = np.hstack(array)

    if smooth_trend:
        win = signal.windows.hann(win_size)
        smoothed_result = signal.convolve(synth, win, mode='same') / sum(win)

        smoothed_result = smoothing(smoothed_result, beta)

#         result = (smoothed_result + result) / 2
        result = (smoothed_result + result) / 2

    return result

def regime_smoothing_FF(synth, real, idx_cp, splitted_real):

    switch_regimes = copy.deepcopy(idx_cp)
    array = []
    for i in range(len(switch_regimes) - 1):
            array.append(synth[switch_regimes[i]:switch_regimes[i+1]])

    # for i in range(len(array) - 1):
    #     delta = array[i][-1] - array[i+1][0]
    #     delta_r = splitted_real[i][-1] - splitted_real[i+1][0]
    #
    #     if (delta >= 0) and (np.sign(delta) == np.sign(delta_r)):
    #
    #         if delta > delta_r:
    #             array[i+1] = array[i+1] + (delta - delta_r)
    #
    #     elif (delta < 0) and (np.sign(delta) == np.sign(delta_r)):
    #
    #         if abs(delta) > abs(delta_r):
    #             array[i+1] = array[i+1] - (abs(delta) - abs(delta_r))
    #
    #     elif (delta >= 0) and (np.sign(delta) != np.sign(delta_r)):
    #         array[i+1] = array[i+1] + (abs(delta) + abs(delta_r))
    #
    #     elif (delta < 0) and (np.sign(delta) != np.sign(delta_r)):
    #         array[i+1] = array[i+1] - (abs(delta) + abs(delta_r))
    #
    #     else:
    #         continue
    for i in range(len(array) - 1):
        delta = array[i][-1] - array[i+1][0]
        delta_r = splitted_real[i][-1] - splitted_real[i+1][0]

        if (delta >= 0) and (np.sign(delta) == np.sign(delta_r)):

            if abs(delta) >= abs(delta_r):
                array[i+1] = array[i+1] + (abs(delta) - abs(delta_r))

            elif abs(delta) < abs(delta_r):
                array[i+1] = array[i+1] - (abs(delta_r) - abs(delta))
            else:
                continue

        elif (delta < 0) and (np.sign(delta) == np.sign(delta_r)):

            if abs(delta) >= abs(delta_r):
                array[i+1] = array[i+1] - (abs(delta) - abs(delta_r))

            elif abs(delta) < abs(delta_r):
                array[i+1] = array[i+1] + (abs(delta_r) - abs(delta))
            else:
                continue

        elif (delta >= 0) and (np.sign(delta) != np.sign(delta_r)):
            array[i+1] = array[i+1] + (abs(delta) + abs(delta_r))

        elif (delta < 0) and (np.sign(delta) != np.sign(delta_r)):
            array[i+1] = array[i+1] - (abs(delta) + abs(delta_r))

        else:
            continue

    result = np.hstack(array)

    return result

