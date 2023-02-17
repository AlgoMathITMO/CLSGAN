import pandas as pd
import numpy as np
import scipy.stats as sts
from scipy.stats import kstest

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn

def m(x):
    y = x.detach().cpu().numpy()
    return torch.FloatTensor(np.concatenate([np.mean(y, axis=1), np.std(y, axis=1),
                sts.skew(y, axis=1), sts.kurtosis(y, axis=1)], axis=1))

def autocorr(x):
    var = x.var()
    mean = x.mean()
    x = x - mean

    result = np.correlate(x, x, mode='full')
    return list(result[len(result)//2:]/var/len(x))

def autocorr_tensor(x, batch_size, seq_len, feature_dim):
    t2 = pd.DataFrame(x.squeeze(2)).apply(lambda x: autocorr(x), axis=1).to_list()
    return torch.FloatTensor(np.array(t2).reshape(batch_size, seq_len, feature_dim))

def spectral_density(x):
    stationarized = pd.Series(x).diff()[1:].diff()[1:].values
    a1 = autocorr(stationarized)
    sd_f = np.fft.fftshift(np.fft.fft(a1))
    mean_asd = np.mean(np.abs(sd_f)**2)

    return mean_asd

def chi(regime: np.array):
    # stationarized = pd.Series(regime).diff()[1:].diff().values[1:]
    # stationarized = pd.Series(regime).diff()[1:].values
    # a1 = autocorr(regime)
    # sd_f = np.fft.fft(a1)
    # mean_asd = np.mean(np.abs(sd_f)**2)
    mean_asd = spectral_density(regime)

    ks = kstest(pd.Series(regime).diff()[1:].values, sts.norm.cdf)[0]

    regime_scaled = (regime - regime.mean()) / regime.std()
    regime_log = np.log(abs(regime_scaled) + 1e-7)

    min_value = regime_log.min()
    max_value = regime_log.max()

#     mean = ((regime - regime.min())/ (regime.max() - regime.min())).mean()
    mean = regime_log.mean()

    return np.array([mean, regime.std(),
                sts.skew(regime), sts.kurtosis(regime), mean_asd, ks, min_value, max_value])
