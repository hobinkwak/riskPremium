import pandas as pd
import numpy as np
from scipy import stats, optimize
from numpy.linalg import svd
from sklearn.metrics.pairwise import pairwise_kernels


def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)  # 외적
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def corr_to_cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


def get_eigen_bound(var, q):
    lb = var * (1 - np.sqrt(q)) ** 2
    ub = var * (1 + np.sqrt(q)) ** 2
    return lb, ub


def get_MP_pdf(x, var, q):
    lb, ub = get_eigen_bound(var, q)
    mask = (x > lb) & (x < ub)
    const = 1 / (2 * np.pi * q * var)
    result = np.zeros(x.shape)
    result[mask] = const * np.sqrt((ub - x[mask]) * (x[mask] - lb)) / x[mask]
    return result


def fit_MP_pdf(q, eigenvalues):
    def objective(var):
        return - np.sum(np.log(1e-9 + get_MP_pdf(eigenvalues, var, q)))

    result = optimize.minimize(objective, x0=np.array([1]),
                               bounds=([1e-9, 2],))
    return result.x[0]


def denoise_covariance(rtn, rm_market=False):
    rtn = rtn - np.nanmean(rtn, axis=0)
    T, n = rtn.shape
    q = n / T
    corr = cov_to_corr(rtn.cov())
    eigval, eigvec = np.linalg.eigh(corr)

    var = fit_MP_pdf(q, eigval)
    lb, ub = get_eigen_bound(var, q)

    if rm_market:
        eigval[(eigval <= ub) & (eigval >= max(eigval))] = np.mean(eigval[eigval <= ub])
    else:
        eigval[eigval <= ub] = np.mean(eigval[eigval <= ub])

    eigval = np.diag(eigval)
    corr_recon = eigvec @ eigval @ eigvec.T
    # set diagonal elements to one by re-weighting eigenvector components
    D = np.diag(np.diag(corr_recon) ** (-1 / 2))
    corr_recon = D @ corr_recon @ D
    cov_recon = corr_to_cov(corr_recon, np.diag(rtn.cov()) ** (1 / 2))
    return cov_recon


def denardCVC(rtns):
    """
    cov_hat = shrinkage_intensity * Target Estimator + (1-shrinakge_intensity) * Covariance

    Target Estimator = phi_hat * Identity Matrix + nu_hat * off-diagonal of Identity matrix

    phi_hat = average of diagonal elements of sample covariance matrix
    nu_hat = average of off diagonal elements of sample covariance matrix

    shrinkage_intensity = min(max(k_hat/T, 0), 1)

    k_hat = (pi_hat - rho_hat) / gamma_hat

    pi_hat = summation of (1/T) sum_{t=1}^T {(r_it - r_bar_i.)*(r_jt - r_bar_j.) - s_ij}^2
    rho_hat = 0 (practically zero, in almost all cases)
    gamma_hat = squared frobenius norm of difference between sample covariance matrix and target estimator)
    """

    if isinstance(rtns, pd.DataFrame):
        rtns = rtns.values
    rtns = rtns.T
    N, T = rtns.shape
    rtns = rtns - np.mean(rtns, axis=1, keepdims=True)

    cov = np.cov(rtns, rowvar=True, ddof=1)

    phi_hat = np.trace(cov) / N  # (1.15a)
    nu_hat = np.mean(cov - np.diag(np.diag(cov)))  # (1.15b)
    target_estimator = np.eye(N) * phi_hat + (1 - np.eye(N)) * nu_hat  # (1.9)

    pi_hat_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            arr1 = rtns[i, :]
            arr2 = rtns[j, :]
            values = []
            for t in range(T):
                values.append(((arr1[t] - arr1.mean()) * (arr2[t] - arr2.mean()) - cov[i, j]) ** 2)
            pi_hat_mat[i, j] = np.mean(np.array(values))
            pi_hat_mat[j, i] = pi_hat_mat[i, j]
    pi_hat = pi_hat_mat.sum()
    gamma_hat = np.square(cov - target_estimator).sum()
    rho = 0
    k_hat = (pi_hat - rho) / gamma_hat
    # shrinkage_estimator = max(min(k_hat/T, 1), 0)
    shrinkage_estimator = min(max(k_hat / T, 0), 1)
    covariance_estimated = (1 - shrinkage_estimator) * cov + shrinkage_estimator * target_estimator
    return covariance_estimated


def determine_n_factor(X, max_k, ic_mode):
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

    # 연산량 줄이기  (논문 상의 scaling constant는 제거 e.g. sqrt(T))
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


class SPCA:
    """
    Ref: Supervised Principal Component Analysis (Ali Ghodsi, 2010)
    """

    def __init__(self, n_component=None, kernel='linear'):
        self.n_component = n_component
        if self.n_component is None:
            self.auto_n = True
        else:
            self.auto_n = False
        self.kernel = kernel

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, np.ndarray):
            y = y.reshape(-1, 1)
        else:
            y = np.array(y).reshape(-1, 1)
        n_obs, n_feat = X.shape
        # X -= X.mean(axis=0)
        # Centetring Matrix  (n_obs, n_obs)
        # equivalent to Demeaning
        H = self._center_matrix(n_obs)
        # Kernelize Y  (n_obs, n_obs)
        L = pairwise_kernels(X=y, metric=self.kernel)
        # (n_feat, n_feat)
        if self.auto_n:
            n_factor = determine_n_factor(H.T @ X, max_k=n_feat, ic_mode=2)
        else:
            n_factor = self.n_component
        Q = X.T.dot(H).dot(L).dot(H).dot(X)
        U, S, Vt = svd(Q)
        self.Q = Q
        self.U = U[:, :n_factor]
        self.S = S[:n_factor]

    def fit_transform(self, X, y):
        self.fit(X, y)
        X_transformed = self.transform(X)
        return X_transformed

    def transform(self, X):
        return X @ self.U

    def _center_matrix(self, n_obs):
        H = np.eye(n_obs) - ((1 / n_obs) * np.ones((n_obs, n_obs)))
        return H


class DirectKernel:
    """
    https://github.com/RefaelLasry/EstimationOfCovarianceMatrix/tree/master
    Implementation of the article:
    "Direct Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices"
    Ledoit and Wolf, Oct 2017,
    translated from authors' Matlab code
    """

    def __init__(self, X):
        self.X = X
        self.n = None
        self.p = None
        self.sample = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.L = None
        self.h = None

    def pav(self, y):
        """
        PAV uses the pair adjacent violators method to produce a monotonic
        smoothing of y
        translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
        """
        y = np.asarray(y)
        assert y.ndim == 1
        n_samples = len(y)
        v = y.copy()
        lvls = np.arange(n_samples)
        lvlsets = np.c_[lvls, lvls]
        flag = 1
        while flag:
            deriv = np.diff(v)
            if np.all(deriv >= 0):
                break

            viol = np.where(deriv < 0)[0]
            start = lvlsets[viol[0], 0]
            last = lvlsets[viol[0] + 1, 1]
            s = 0
            n = last - start + 1
            for i in range(start, last + 1):
                s += v[i]

            val = s / n
            for i in range(start, last + 1):
                v[i] = val
                lvlsets[i, 0] = start
                lvlsets[i, 1] = last
        return v

    def estimate_cov_matrix(self):

        # extract sample eigenvalues sorted in ascending order and eigenvectors
        self.n, self.p = self.X.shape
        self.sample = (self.X.transpose() @ self.X) / self.n
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.sample)
        isort = np.argsort(self.eigenvalues, axis=-1)
        self.eigenvalues.sort()
        self.eigenvectors = self.eigenvectors[:, isort]

        # compute direct kernel estimator
        self.eigenvalues = self.eigenvalues[max(1, self.p - self.n + 1) - 1:self.p]
        self.L = np.repeat(self.eigenvalues, min(self.n, self.p), axis=0).reshape(self.eigenvalues.shape[0],
                                                                                  min(self.n, self.p))
        self.h = self.n ** (-0.35)
        component_00 = 4 * (self.L.T ** 2) * self.h ** 2 - (self.L - self.L.T) ** 2
        component_0 = np.maximum(np.zeros((component_00.shape[1], component_00.shape[1])), component_00)
        component_a = np.sqrt(component_0)
        component_b = 2 * np.pi * (self.L.T ** 2) * self.h ** 2
        ftilda = np.mean(component_a / component_b, axis=1)

        com_1 = np.sign(self.L - self.L.T)
        com_2_1 = (self.L - self.L.T) ** 2 - 4 * self.L.T ** 2 * self.h ** 2
        com_2 = np.maximum(np.zeros((com_2_1.shape[1], com_2_1.shape[1])), com_2_1)
        com_3_1 = np.sqrt(com_2)
        com_3_2 = com_1 * com_3_1
        com_3 = com_3_2 - self.L + self.L.T
        com_4 = 2 * np.pi * self.L.T ** 2 * self.h ** 2
        com_5 = com_3 / com_4
        Hftilda = np.mean(com_5, axis=1)

        if self.p <= self.n:
            com_0 = (np.pi * (self.p / self.n) * self.eigenvalues * ftilda) ** 2
            com_1 = (1 - (self.p / self.n) - np.pi * (self.p / self.n) * self.eigenvalues * Hftilda) ** 2
            com_2 = com_0 + com_1
            dtilde = self.eigenvalues / com_2
        else:
            Hftilda0 = (1 - np.sqrt(max(1 - 4 * self.h ** 2, 0))) / (2 * np.pi * self.n * self.h ** 2) * np.mean(
                1 / self.eigenvalues)
            dtilde0 = 1 / (np.pi * ((self.p - self.n) / self.n) * Hftilda0)
            dtilde1 = self.eigenvalues / np.pi ** 2 * self.eigenvalues ** 2 * (ftilda ** 2 + Hftilda ** 2)
            dtilde = np.hstack((dtilde0 * np.ones((self.p - self.n, 1)).reshape(self.p - self.n, ), dtilde1))

        dhat = self.pav(dtilde)
        sigmahat = np.dot(self.eigenvectors, (np.tile(dhat, (self.p, 1)).T * self.eigenvectors.T))
        return sigmahat
