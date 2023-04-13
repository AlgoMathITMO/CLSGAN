import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import sys
# sys.path.insert(1, '/Synth time series Project/tools/')
from tools.data_preprocessing import real_data_loading, split_ts
import models.TCBGAN as gan
from tools.data_preprocessing import *
from models.GAN_models import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def GAN_SSA_training(time_series, epochs=50, seq_len=2**7, batch_size=2**5, n_fclayers=3, feature_dim=1, z_dim=3,
                                    wsize=2**7, groups=2, logreturn=True, lambert=True, decomposition=False):
    if decomposition:
        processed_params = ts_preprocessing(time_series, wsize, groups, logreturn, lambert, decomposition)
        ts_processed = processed_params[:, 0]
        ts_processed = np.array([np.hstack(ts_processed)])
        
        
        generators = []
        i = 0
        for ts in ts_processed:
            # print(f'---------------------------\nStart training SSA generator {i}\n---------------------------')
            generator = gan.train(ts.reshape(len(ts), 1), epochs, seq_len, batch_size, n_fclayers, feature_dim, z_dim)
            generators.append(generator)
            i += 1

        return generators, processed_params

    else:
        processed_params = ts_preprocessing(time_series, wsize, groups, logreturn, lambert, decomposition)
        ts_processed = processed_params[0]
        generator = gan.train(ts_processed.reshape(len(ts_processed), 1), epochs, seq_len, batch_size, n_fclayers, feature_dim, z_dim)
        return generator, processed_params

def QuantGAN_SSA_training(time_series, epochs=20, seq_len=2**7, batch_size=2**5, n_fclayers=3, feature_dim=1, z_dim=3,
                                    wsize=2**6, groups=2, logreturn=True, lambert=True, decomposition=False):
    if decomposition:
#         decomposition = SingularSpectrumAnalysis(window_size=wsize, groups=groups)
        log_returns_preprocessed, log_returns, standardScaler1, standardScaler2, gaussianize = quantgan_preprocessing(pd.Series(time_series))

#         X_ssa = decomposition.fit_transform(log_returns_preprocessed.reshape(1, -1))
#         ts_processed = np.array([np.hstack(X_ssa)])
        
        X_decomposed = [log_returns_preprocessed.T[0]]
        for i in range(groups):
            X_decomposed.append(log_returns_preprocessed.T[0] + np.random.normal(0, 0.1, size=len(log_returns_preprocessed.T[0])))
        ts_processed = np.array([np.hstack(X_decomposed)])

        generators = []
        i = 0
        for ts in ts_processed:
            # print(f'---------------------------\nStart training SSA generator {i}\n---------------------------')
            generator, scores = quantgan_fit(ts, epochs, seq_len, batch_size, z_dim, train=True)
            generators.append(generator)
            i += 1

        return generators, log_returns, standardScaler1, standardScaler2, gaussianize, scores

    else:
        log_returns_preprocessed, log_returns, standardScaler1, standardScaler2, gaussianize = quantgan_preprocessing(pd.Series(time_series))
        generator, scores = quantgan_fit(log_returns_preprocessed, epochs, seq_len, batch_size, z_dim, train=True)
        return generator, log_returns, standardScaler1, standardScaler2, gaussianize, scores

def CLSGAN_training(time_series, epochs=50, seq_len=2**7, batch_size=2**5, n_fclayers=3, feature_dim=1, z_dim=3,
                                    wsize=2**6, groups=2, decomposition=False):
    if decomposition:
#         decomposition = SingularSpectrumAnalysis(window_size=wsize, groups=groups)
        log_returns_preprocessed, log_returns, standardScaler1,\
            standardScaler2, gaussianize, m = clsgan_preprocessing(pd.Series(time_series))

#         X_ssa = decomposition.fit_transform(log_returns_preprocessed.reshape(1, -1))
        X_decomposed = [log_returns_preprocessed.T[0]]
        for i in range(groups):
            X_decomposed.append(log_returns_preprocessed.T[0] + np.random.normal(0, 0.2, size=len(log_returns_preprocessed.T[0])))
        ts_processed = np.array([np.hstack(X_decomposed)])

        generators = []
        supervisors = []
        i = 0
        for ts in ts_processed:
            # print(f'---------------------------\nStart training SSA generator {i}\n---------------------------')
            generator, supervisor, scores = clsgan_fit(ts, epochs, seq_len, batch_size, z_dim, train=True)
            generators.append(generator)
            supervisors.append(supervisor)

            i += 1

        return generators, supervisors, log_returns, standardScaler1, standardScaler2, gaussianize, m, scores

    else:
        log_returns_preprocessed, log_returns, standardScaler1,\
            standardScaler2, gaussianize, m = clsgan_preprocessing(pd.Series(time_series))

        generator, supervisor, scores = clsgan_fit(log_returns_preprocessed, epochs, seq_len, batch_size, z_dim, train=True)
        return generator, supervisor, log_returns, standardScaler1, standardScaler2, gaussianize, m, scores

def generate_synth(ts, generators, processed_params, z_dim=3, decomposition=False):
    if decomposition:
        z = torch.randn(1, len(ts), z_dim, device=device)
        F_features = []
        for i in range(len(generators)):
            fake = generators[i](z).detach().cpu().reshape(len(ts)).numpy()
            ts_inverse_lambert = inverse(fake * processed_params[i][2], processed_params[i][1]) + processed_params[i][3]
            # ts_inverse_lambert = inverse(fake * processed_params[i][2], processed_params[i][1])
            #if use scaler
            # ts_inverse_lambert = processed_params[i][1][3][0].inverse_transform(ts_inverse_lambert.reshape(-1,1)).T[0]

            F_features.append(ts_inverse_lambert)

        F_features = np.array(F_features)
        F_tilde = F_features
        # F_tilde = np.sum(F_features, axis=0)

        return F_tilde

    else:
        z = torch.randn(1, len(ts), z_dim, device=device)
        fake = generators(z).detach().cpu().reshape(len(ts)).numpy()
        ts_inverse_lambert = inverse(fake * processed_params[2], processed_params[1]) + processed_params[3]
        # ts_inverse_lambert = inverse(fake * processed_params[2], processed_params[1])
        #if use scaler
        # ts_inverse_lambert = processed_params[1][3][0].inverse_transform(ts_inverse_lambert.reshape(-1,1)).T[0]

        F_tilde = ts_inverse_lambert

        return F_tilde

def QuantGAN_generate_synth(ts, generators, processed_params, z_dim=3, decomposition=False):
    if decomposition:
        # z = torch.randn(1, len(ts), z_dim, device=device)
        F_features = []
        for i in range(len(generators)):
            res = quantGAN_generate_synth(generators[i], processed_params[0], ts, processed_params[1], processed_params[2], processed_params[3], z_dim)

            F_features.append(res)

        F_features = np.array(F_features)
        F_tilde = F_features
        # F_tilde = np.sum(F_features, axis=0)
        return F_tilde

    else:
        res = quantGAN_generate_synth(generators, processed_params[0], ts, processed_params[1], processed_params[2], processed_params[3], z_dim)
        return res

def CLSGAN_generate_synth(ts, generators, supervisors, processed_params, z_dim=3, decomposition=False):
    if decomposition:
        # z = torch.randn(1, len(ts), z_dim, device=device)
        F_features = []
        for i in range(len(generators)):
            res = generate_sample_CLSGAN(generators[i], supervisors[i], processed_params[0],\
                ts, processed_params[1], processed_params[2], processed_params[3], processed_params[4], z_dim)

            F_features.append(res)

        F_features = np.array(F_features)
        F_tilde = F_features
        # F_tilde = np.sum(F_features, axis=0)
        return F_tilde

    else:
        res = generate_sample_CLSGAN(generators, supervisors, processed_params[0],\
            ts, processed_params[1], processed_params[2], processed_params[3], processed_params[4], z_dim)
        return res
