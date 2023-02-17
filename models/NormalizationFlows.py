from models.utils.spectral import get_frequencies, power_spectral_density
from models.SequentialFlows import FourierFlow, RealNVP

import pandas as pd
import os
import random
import numpy as np
import torch

def f_fit(num_clusters, ts_wind, epochs=1000, hidden=200, n_flows=10, display_step=100, lr=1e-4):
    FF_models = []
    # RVP_models = []

    for j in range(num_clusters):
        X = ts_wind[j]
        X = np.log(pd.Series(X) / pd.Series(X).shift(1))[1:].values

        X = X[:(len(X) // 4 * 4 + 1 if len(X) % 4 > 0 else len(X) - 3)]

        T = len(X)
        X = [X]

        FF_model    = FourierFlow(hidden=hidden, fft_size=T, n_flows=n_flows, normalize=False)
        # RVP_model   = RealNVP(hidden=hidden, T=T, n_flows=n_flows)

        print('Start training Fourier Flows')
        FF_losses   = FF_model.fit(X, epochs=epochs, batch_size=2**7,
                           learning_rate=lr, display_step=display_step)

        # print('Start training RealNVP')
        # RVP_losses  = RVP_model.fit(X, epochs=epochs, batch_size=2**7,
        #                     learning_rate=lr, display_step=display_step)

        FF_models.append(FF_model)
        # RVP_models.append(RVP_model)

    # return FF_models, RVP_models
    return FF_models

def synths_flows(FF_models, num_clusters, init_values, n_samples=1):
    synths_FF = []
    # synths_RVP = []

    for j in range(num_clusters):
        X_gen_FF   = FF_models[j].sample(n_samples)
        # X_gen_RVP  = RVP_models[j].sample(n_samples)

        synths_FF.append(init_values[j] * np.exp(X_gen_FF.cumsum()))
        # synths_RVP.append(init_values[j] * np.exp(X_gen_RVP.cumsum()))

    return synths_FF
