import pandas as pd
import numpy as np
from numpy.linalg import svd


def determine_n_factor(X, max_k, ic_mode):
    """
    Determining the Number of Factors in Approximate Factor Models (2003, Jushan Bai, Serena Ng)
    """
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    T, N = X.shape
    NT = N * T

    ks = np.arange(1, max_k + 1)
    a = min(N, T)

    # Calculate penalty
    if ic_mode == 1:
        overfit_penalty = np.log(NT / (N + T)) * ks * ((N + T) / NT)
    elif ic_mode == 2:
        overfit_penalty = np.log(a) * ks * ((N + T) / NT)
    elif ic_mode == 3:
        overfit_penalty = np.log(a) / a * ks

    # reduce computational workload
    # the scaling constant in the paper is removed   e.g. sqrt(T)
    large_T = T >= N
    if large_T:
        U, S, VT = np.linalg.svd(X.T @ X)
    else:
        U, S, VT = np.linalg.svd(X @ X.T)

    # select number of factors
    V = np.zeros(max_k + 1)
    IC1 = np.zeros(max_k + 1)

    for i in range(max_k):
        U_sub = U[:, :i + 1]
        c_hat = (X @ U_sub @ U_sub.T) if large_T else (U_sub @ U_sub.T @ X)
        resid = X - c_hat
        V[i] = ((resid * resid / T).sum(axis=0)).mean()  # MSE
        IC1[i] = np.log(V[i]) + overfit_penalty[i]

    V[max_k] = (X * X / T).sum(axis=0).mean()
    IC1[max_k] = np.log(V[max_k])  # value of the information criterion when using no factors

    ic_min_idx = IC1.argmin(axis=0)
    optim_k = ic_min_idx + 1

    return optim_k
