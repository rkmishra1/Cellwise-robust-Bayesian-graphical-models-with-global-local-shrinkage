"""
Frequentist competitors for cellwise- and casewise-robust precision estimation.

All return an estimated precision matrix.

  glasso        sample covariance + graphical lasso                     (non-robust baseline)
  npn           nonparanormal SKEPTIC (Kendall tau) + glasso            (robust to monotone marginals)
  tyler         Tyler M-estimator of scatter + glasso                   (casewise robust)
  ddc_glasso    DDC-style cellwise filter -> impute -> glasso           (cellwise robust)

HONESTY NOTE.  `ddc_glasso` is a simplified reimplementation of the DetectDeviatingCells idea of
Rousseeuw and Van den Bossche (2018) -- robust standardisation, prediction of each cell from its
correlated neighbours, flagging of cells whose standardised residual is large, imputation, then a
graphical lasso.  It is NOT the published DDC algorithm and should be labelled as such in any
write-up.  Likewise no attempt is made here to reproduce cellMCD (Raymaekers and Rousseeuw, 2022)
or 2SGS (Agostinelli et al., 2015); those require the authors' implementations and remain a gap.
"""

import numpy as np
from scipy.stats import kendalltau, norm
from sklearn.covariance import graphical_lasso


# ------------------------------------------------------------------ helpers

def _to_corr(S):
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return S / np.outer(d, d), d


def _pd_project(S, eps=1e-3):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    return V @ np.diag(w) @ V.T


def _glasso_bic(S, n, alphas=(0.02, 0.05, 0.1, 0.2, 0.35)):
    """Graphical lasso with alpha chosen by BIC on the correlation scale."""
    R, d = _to_corr(_pd_project(S))
    best, best_bic = None, np.inf
    for a in alphas:
        try:
            _, prec = graphical_lasso(R, alpha=a, max_iter=150)
        except Exception:
            continue
        sign, logdet = np.linalg.slogdet(prec)
        if sign <= 0:
            continue
        k = int((np.abs(prec) > 1e-8).sum() - prec.shape[0]) // 2
        bic = -n * (logdet - np.trace(R @ prec)) + k * np.log(n)
        if bic < best_bic:
            best_bic, best = bic, prec
    if best is None:
        best = np.linalg.inv(R + 0.1 * np.eye(R.shape[0]))
    return best / np.outer(d, d)          # back to the original scale


# ------------------------------------------------------------------ estimators

def glasso(X):
    Xc = X - X.mean(0)
    return _glasso_bic(np.cov(Xc, rowvar=False), X.shape[0])


def npn(X):
    """Nonparanormal SKEPTIC: Sigma_jk = sin(pi/2 * Kendall tau_jk)."""
    n, p = X.shape
    T = np.eye(p)
    for j in range(p):
        for k in range(j + 1, p):
            t = kendalltau(X[:, j], X[:, k]).statistic
            T[j, k] = T[k, j] = np.sin(np.pi / 2 * (0.0 if np.isnan(t) else t))
    return _glasso_bic(_pd_project(T), n)


def tyler(X, iters=100, tol=1e-6):
    """Tyler's M-estimator of scatter (casewise robust, distribution-free in the elliptical model)."""
    n, p = X.shape
    Xc = X - np.median(X, axis=0)
    V = np.eye(p)
    for _ in range(iters):
        Vi = np.linalg.inv(V)
        q = np.einsum('ij,jk,ik->i', Xc, Vi, Xc)
        q = np.clip(q, 1e-12, None)
        Vn = (p / n) * (Xc / q[:, None]).T @ Xc
        Vn = Vn / np.trace(Vn) * p
        if np.abs(Vn - V).max() < tol:
            V = Vn
            break
        V = Vn
    # restore scale with a robust univariate scale per coordinate
    s = 1.4826 * np.median(np.abs(Xc), axis=0)
    Vc, _ = _to_corr(V)
    return _glasso_bic(Vc * np.outer(s, s), n)


def ddc_filter(X, corr_thresh=0.5, cutoff=None):
    """Simplified DDC-style cellwise filter.  Returns (imputed X, flag mask)."""
    n, p = X.shape
    if cutoff is None:
        cutoff = norm.ppf(0.995)
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    mad = np.where(mad < 1e-8, 1.0, mad)
    Z = (X - med) / mad
    Zw = np.clip(Z, -3.0, 3.0)                      # wrap for robust correlation

    C = np.corrcoef(Zw, rowvar=False)
    C = np.nan_to_num(C)
    np.fill_diagonal(C, 0.0)

    pred = np.zeros_like(Z)
    for j in range(p):
        nb = np.where(np.abs(C[:, j]) >= corr_thresh)[0]
        if nb.size == 0:
            continue
        w = np.abs(C[nb, j])
        # robust slope of Z_j on Z_k is approximately C[k,j] for standardised columns
        contrib = Zw[:, nb] * C[nb, j][None, :]
        pred[:, j] = (contrib * w[None, :]).sum(1) / w.sum()

    resid = Z - pred
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid, axis=0)), axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    flag = np.abs(resid / scale) > cutoff

    Zi = np.where(flag, pred, Z)
    return Zi * mad + med, flag


def ddc_glasso(X):
    Xi, _ = ddc_filter(X)
    Xc = Xi - Xi.mean(0)
    return _glasso_bic(np.cov(Xc, rowvar=False), X.shape[0])


def cellmcd_glasso(X):
    from cellwise import cellmcd
    _, S, _ = cellmcd(X)
    return _glasso_bic(_pd_project(S), X.shape[0])


def twosgs_glasso(X):
    from cellwise import twosgs
    _, S, _ = twosgs(X)
    return _glasso_bic(_pd_project(S), X.shape[0])


METHODS = {'glasso': glasso, 'npn': npn, 'tyler': tyler, 'ddc+glasso': ddc_glasso,
           'cellMCD+glasso': cellmcd_glasso, '2SGS+glasso': twosgs_glasso}
