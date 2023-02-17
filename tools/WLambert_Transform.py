import numpy as np
from scipy.optimize import fmin
from scipy.special import lambertw
from scipy.stats import kurtosis, norm
from sklearn.preprocessing import StandardScaler
import warnings

# def w_d(z, delta):
#     # Eq. 9
#     if delta < 1e-6:
#         return z
#     return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)
#
#
# def w_t(y, tau):
#     # Eq. 8
#     return tau[0] + tau[1] * w_d((y - tau[0]) / tau[1], tau[2])
#
#
# def inverse(x, tau):
#     # Eq. 6
#     u = (x - tau[0]) / tau[1]
#     return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))
#
#
# def igmm(y: np.ndarray, tol: float = 1e-6, max_iter: int = 100):
#     # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
#     if np.std(y) < tol:
#         return [np.mean(y), np.clip(np.std(y), -tol, tol), 0]
#     delta0 = delta_init(y)
#     tau1 = [np.median(y), np.std(y) * (1. - 2. * delta0) ** 0.75, delta0]
#     for k in range(max_iter):
#         tau0 = tau1
#         z = (y - tau1[0]) / tau1[1]
#         delta1 = delta_gmm(z)
#         x = tau0[0] + tau1[1] * w_d(z, delta1)
#         mu1, sigma1 = np.mean(x), np.std(x)
#         tau1 = [mu1, sigma1, delta1]
#
#         if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
#             break
#         else:
#             if k == max_iter - 1:
#                 print("Lambert Warning: No convergence after %d iterations. Increase max_iter." % max_iter)
#     return tau1
#
#
# def delta_gmm(z):
#     # Alg. 1, Appendix C
#     delta0 = delta_init(z)
#
#     def func(q):
#         u = w_d(z, np.exp(q))
#         if not np.all(np.isfinite(u)):
#             return 0.
#         else:
#             k = kurtosis(u, fisher=True, bias=False)**2
#             if not np.isfinite(k) or k > 1e10:
#                 return 1e10
#             else:
#                 return k
#
#     res = fmin(func, np.log(delta0), disp=0)
#     return np.around(np.exp(res[-1]), 6)
#
#
# def delta_init(z):
#     gamma = kurtosis(z, fisher=False, bias=False)
#     with np.errstate(all='ignore'):
#         delta0 = np.clip(1. / 66 * (np.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.48)
#     if not np.isfinite(delta0):
#         delta0 = 0.01
#     return delta0
#
# def data_transform(data):
#     # scaler1 = StandardScaler()
#     # data_norm = (scaler1.fit_transform(data.reshape(-1, 1))).T[0]
#
#     data_mean = np.mean(data)
#     data_norm = data - data_mean
#     params = igmm(data_norm)
#     data_processed = w_d((data_norm - params[0]) / params[1], params[2])
#     data_max = np.max(np.abs(data_processed))
#     data_processed /= data_max
#
#     # params.append([scaler1])
#
#     return data_processed, params, data_max, data_mean
# import numpy as np
# from scipy.optimize import fmin
# from scipy.special import lambertw
# from scipy.stats import kurtosis, norm
# from sklearn.preprocessing import StandardScaler
#
def delta_init(z):
    k = kurtosis(z, fisher=False, bias=False)
    if k < 166. / 62.:
        return 0.01

    return np.clip(1. / 66 * (np.sqrt(66 * k - 162.) - 6.), 0.01, 0.48)

def delta_gmm(z):
    delta = delta_init(z)

    def iter(q):
        u = W_delta(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.
        k = kurtosis(u, fisher=True, bias=False)**2
        if not np.isfinite(k) or k > 1e10:
            return 1e10
        return k

    res = fmin(iter, np.log(delta), disp=0)
    return np.around(np.exp(res[-1]), 6)

def W_delta(z, delta):
    return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)

def W_params(z, params):
    return params[0] + params[1] * W_delta((z - params[0]) / params[1], params[2])

def inverse(z, params):
    return params[0] + params[1] * (z * np.exp(z * z * (params[2] * 0.5)))

def igmm(z, eps=1e-6, max_iter=100):
    delta = delta_init(z)
    params = [np.median(z), np.std(z) * (1. - 2. * delta) ** 0.75, delta]
    for k in range(max_iter):
        params_old = params
        u = (z - params[0]) / params[1]
        params[2] = delta_gmm(u)
        x = W_params(z, params)
        params[0], params[1] = np.mean(x), np.std(x)

        if np.linalg.norm(np.array(params) - np.array(params_old)) < eps:
            break
        # if k == max_iter - 1:
        #     print("Solution not found")

    return params

def data_transform(data):
    # scaler1 = StandardScaler()
    # data_norm = (scaler1.fit_transform(data.reshape(-1, 1))).T[0]

    data_mean = np.mean(data)
    data_norm = data - data_mean
    params = igmm(data_norm)
    data_processed = W_delta((data_norm - params[0]) / params[1], params[2])
    data_max = np.max(np.abs(data_processed))
    data_processed /= data_max

    # params.append([scaler1])

    return data_processed, params, data_max, data_mean
