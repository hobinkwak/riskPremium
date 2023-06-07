import pandas as pd
import numpy as np
from utils import *
import statsmodels.api as sm


class Estimator:

    def __init__(self, rtn: pd.DataFrame, macro_factor: pd.DataFrame):

        self.rtn = rtn
        self.macro_factor = macro_factor
        self.macro_index = macro_factor.index
        self.macro_cols = macro_factor.columns

    def three_pass(self, max_k=None, return_fmp=False):
        """
        Asset Pricing with omitted factors
        """
        if max_k is None:
            max_k = self.rtn.shape[-1] - 1
        assert max_k < self.rtn.shape[-1], "max_k는 return의 columns size보다 작게"
        rtn = self.rtn.values.T
        macro = self.macro_factor.values.T

        rtn_dm, r_bar = self._demean(rtn)
        macro_dm, _ = self._demean(macro)

        self._get_p(rtn_dm, max_k)
        V = self._get_V(rtn_dm)  # p x T
        beta = self._get_beta(rtn_dm, V)  # p x n
        gamma = self._get_gamma(r_bar, beta)  # p
        eta = self._get_eta(macro_dm, V)  # d x p
        # G: d x T
        risk_premia, G = self._three_pass_risk_premia(V, eta, gamma)
        se = self._three_pass_standard_error(V, macro_dm, eta, gamma)
        t_stat = risk_premia / se
        r2g = self._get_r2g(macro_dm, eta, V)
        result = pd.DataFrame([t_stat, risk_premia, se, r2g],
                              columns=self.macro_cols, index=['t-stat', 'risk premia', 's.e.', 'R2g'])

        if return_fmp:
            fmp = pd.DataFrame(G.T, index=self.macro_index, columns=self.macro_cols)
            return result, fmp
        return result

    def _demean(self, rtn):
        r_bar = np.mean(rtn, axis=1).reshape(-1, 1)
        rtn_dm = rtn - r_bar
        return rtn_dm, r_bar

    def _get_p(self, rtn_dm, max_k):
        n, T = rtn_dm.shape
        p = determine_n_factor(rtn_dm / np.sqrt(n) / np.sqrt(T), max_k=max_k, ic_mode=2)
        self.p = p

    def _get_V(self, rtn_dm):
        n, T = rtn_dm.shape
        U, S, Vt = np.linalg.svd(rtn_dm / np.sqrt(n) / np.sqrt(T), full_matrices=False)
        V = Vt.T  # (T, n_macro)
        V_hat = T ** 0.5 * V[:, :self.p].T  # (p, T)
        V_hat = V_hat / np.linalg.norm(V_hat, axis=1).reshape(-1, 1)
        return V_hat

    def _get_beta(self, rtn_dm, V):
        res = sm.OLS(rtn_dm.T, sm.add_constant(V.T)).fit()
        beta = res.params[1:, :]  # p x n
        return beta

    def _get_gamma(self, r_bar, beta):
        res = sm.OLS(r_bar, beta.T).fit()  # no intercept
        gamma = res.params  # p
        return gamma

    def _get_eta(self, macro_dm, V):
        res = sm.OLS(macro_dm.T, sm.add_constant(V.T)).fit()
        if res.params.ndim == 1:
            res.params = res.params.reshape(1, -1)
            eta = res.params[:, 1:]  # d x p
        else:
            eta = res.params[1:, :].T
        return eta

    def _get_r2g(self, macro_dm, eta, V):
        return (1 / np.diag(macro_dm @ macro_dm.T)) * np.diag(eta @ (V @ V.T) @ eta.T)

    def _three_pass_risk_premia(self, V, eta, gamma):
        G = eta @ V  # fitted value of macro factor  (d, t)
        risk_premia = eta @ gamma
        return risk_premia, G

    def _vec(self, arr):
        return arr.reshape((-1, 1), order='F')

    def _three_pass_standard_error(self, V, macro_dm: np.ndarray, eta, gamma, q=4):
        """
        V: p x T
        eta: d x p
        gamma: p
        """
        Sigma_v = V @ V.T * (1 / V.shape[1])  # p x p
        Z = (macro_dm - eta @ V)  # d x T
        d, T = Z.shape

        p = V.shape[0]
        Pi11 = np.zeros((d * p, d * p))
        for i in range(T):
            a = self._vec(Z[:, i].reshape(-1, 1) @ V[:, i].reshape(-1, 1).T)
            Pi11 += a @ a.T / T
        for i in range(q):
            for j in range(i + 1, T):
                a = self._vec(Z[:, j - i].reshape(-1, 1) @ V[:, j - i].reshape(-1, 1).T)
                b = self._vec((Z[:, j].reshape(-1, 1) @ V[:, j].reshape(-1, 1).T))
                Pi11 += (1 - (i + 1) / 5) * (a @ b.T + b @ a.T) / T

        Pi12 = np.zeros((d * p, p))
        for i in range(T):
            a = self._vec(Z[:, i].reshape(-1, 1) @ V[:, i].reshape(-1, 1).T)
            Pi12 += a @ V[:, i].reshape(-1, 1).T / T
        for i in range(q):
            for j in range(i + 1, T):
                a = self._vec(Z[:, j - i].reshape(-1, 1) @ V[:, j - i].reshape(-1, 1).T)
                b = self._vec(Z[:, j].reshape(-1, 1) @ V[:, j].reshape(-1, 1).T)
                Pi12 += (1 - (i + 1) / 5) * (a @ V[:, j].reshape(-1, 1).T + b @ V[:, j - i].reshape(-1, 1).T) / T

        Pi22 = np.zeros((p, p))
        for i in range(T):
            Pi22 += V[:, i].reshape(-1, 1) @ V[:, i].reshape(-1, 1).T / T
        for i in range(q):
            for j in range(i + 1, T):
                Pi22 += (1 - (i + 1) / 5) * (
                        V[:, j - i].reshape(-1, 1) @ V[:, j].reshape(-1, 1).T + V[:, j].reshape(-1, 1) @ V[:,
                                                                                                         j - i].reshape(
                    -1, 1).T) / T

        mat1 = np.kron(a=gamma.reshape(-1, 1).T @ np.linalg.inv(Sigma_v),
                       b=np.identity(d))
        mat2 = np.kron(a=np.linalg.inv(Sigma_v) @ gamma.reshape(-1, 1),
                       b=np.identity(d))
        se = np.diag((mat1 @ Pi11 @ mat2) / T + (mat1 @ Pi12 @ eta.T) / T + (mat1 @ Pi12 @ eta.T).T / T + (
                eta @ Pi22 @ eta.T) / T)

        se = np.sqrt(se)
        return se

    def two_pass(self, adjust_autocorr=True):
        """
        Fama-Macbeth style
        Fama MacBeth regressions provide standard errors corrected only for cross-sectional correlation.
        The standard errors from this method do not correct for time-series autocorrelation.
        This is usually not a problem for stock trading since stocks have weak time-series autocorrelation
        in daily and weekly holding periods, but autocorrelation is stronger over long horizons.
        This means Fama MacBeth regressions may be inappropriate to use in many corporate finance settings
        where project holding periods tend to be long.
        For alternative methods of correcting standard errors for time series and cross-sectional correlation
        in the error term look into double clustering by firm and year
        """
        rtn = self.rtn.values
        macro_factor = self.macro_factor.values
        res = sm.OLS(rtn, sm.add_constant(macro_factor)).fit()
        betas = res.params[1:].T
        lbds = []
        for t in range(rtn.shape[0]):
            res = sm.OLS(rtn[t, :],
                         betas).fit()
            lbds.append(res.params)
        lbds = pd.DataFrame(lbds, columns=self.macro_cols)

        result = []
        for factor in lbds.columns:
            if adjust_autocorr:
                res = sm.OLS(lbds[factor], np.ones(len(lbds[factor]))).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            else:
                res = sm.OLS(lbds[factor], np.ones(len(lbds[factor]))).fit()
            result.append([res.tvalues.values[0], res.params.values[0], res.bse.values[0]])
        result = pd.DataFrame(result, index=self.macro_cols, columns=['t-stat', 'risk premia', 's.e.']).T
        return result


if __name__ == '__main__':
    pass
