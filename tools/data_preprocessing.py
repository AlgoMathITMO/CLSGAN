import numpy as np
import pandas as pd
import stumpy
from stumpy.floss import _cac
import copy
from pyts.decomposition import SingularSpectrumAnalysis
from tools.WLambert_Transform import *

def real_data_loading(data: np.array, seq_len):
    ori_data = data[::]
    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len + 1):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data


def real_data_loading_index(data: np.array, seq_len):
    ori_data = data[::]
    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len + 1):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data, idx

def real_data_loading_permute(data: np.array, seq_len, idx):
    ori_data = data[::]
    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len + 1):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data

def split_ts(ts):
    m = 2**8
    L = 2**8

    mp = stumpy.stump(ts, m=m)
    cac, regime_locations = stumpy.fluss(mp[:, 1], L=L, n_regimes=5, excl_factor=1)
    points = sorted(regime_locations)

    switch_regimes = copy.deepcopy(points)
    switch_regimes.append(len(ts))
    switch_regimes.insert(0, 0)

    array = []

    for i in range(len(switch_regimes) - 1):
        array.append(ts[switch_regimes[i]:switch_regimes[i+1]])

    return np.array(array)

# def cond_vector(x: np.array, seq_len):
#     array = []
#     for item in x:
#         cond = np.repeat(np.array([pd.DataFrame(item).describe().values[1:2].flatten()]), seq_len, axis=0)
#         array.append(cond)
#     return array

# def time_steps_preprocessing(data: np.array, seq_len):
#     ori_data = data[::]
#
#     temp_data = []
#     cond_data = []
#     # Cut data by sequence length
#     for i in range(0, len(ori_data) - seq_len + 1):
#         _x = ori_data[i:i + seq_len]
#         cond = np.repeat(np.array([pd.DataFrame(_x).describe().values[1:2].flatten()]), seq_len, axis=0)
# #         temp_data.append(np.concatenate((_x, cond), axis=1))
#         temp_data.append(_x)
#         cond_data.append(cond)
#
#     # Mix the datasets (to make it similar to i.i.d)
#     idx = np.random.permutation(len(temp_data))
#     data = []
#     cond_vector = []
#     for i in range(len(temp_data)):
#         data.append(temp_data[idx[i]])
#         cond_vector.append(cond_data[idx[i]])
#
#     return torch.cat([torch.FloatTensor(data), torch.FloatTensor(cond_vector)], axis=2)

def ts_preprocessing(ts: np.array, wsize=2**7, groups=10, logreturn=True, lambert=True, ssa=True):
    if logreturn:
        ts = pd.DataFrame(ts)
        ts = np.log(ts / ts.shift(1))[1:].values

        if ssa:
            ssa = SingularSpectrumAnalysis(window_size=wsize, groups=groups)
            X_ssa = ssa.fit_transform(ts.reshape(1, -1))

            if lambert:
                lambert_params = []
                for item in X_ssa:
                    data_processed, params, data_max, data_mean = data_transform(item)
                    lambert_params.append((data_processed, params, data_max, data_mean))

                lambert_params = np.array(lambert_params)

                return lambert_params
            else:
                return X_ssa
        else:
            if lambert:
                data_processed, params, data_max, data_mean = data_transform(ts)
                lambert_params = np.array([data_processed, params, data_max, data_mean])
                return lambert_params

            else:
                return ts

    else:
        if ssa:
            ssa = SingularSpectrumAnalysis(window_size=wsize, groups=groups)
            X_ssa = ssa.fit_transform(ts.reshape(1, -1))

            if lambert:
                lambert_params = []
                for item in X_ssa:
                    data_processed, params, data_max, data_mean = data_transform(item)
                    lambert_params.append((data_processed, params, data_max, data_mean))

                lambert_params = np.array(lambert_params)

                return lambert_params
            else:
                return X_ssa
        else:
            if lambert:
                data_processed, params, data_max, data_mean = data_transform(ts)
                lambert_params = np.array([data_processed, params, data_max, data_mean])
                return lambert_params

            else:
                return ts
