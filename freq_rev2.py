"""Frequentist side of the enlarged comparison and of the fault-injection example.

Each covariance estimator is passed to the graphical lasso, with the penalty chosen either by
BIC or by 3-fold cross-validation of the Gaussian log-likelihood, both over the same 12-point grid.
For the cellwise pipelines the folds are taken from the method's cleaned (imputed) data matrix,
because cellMCD and GSE are not reliable on folds with n close to p.

Usage: python freq_rev2.py export | r | score
"""
import json, os, subprocess, sys, warnings
import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.metrics import roc_auc_score
import exp_rev2 as E

warnings.filterwarnings('ignore')
DIN, DOUT, OUT = 'rev2_h2h_data', 'rev2_h2h_out', 'rev2_freq.jsonl'
LAMBDAS = np.exp(np.linspace(np.log(0.01), np.log(0.6), 12))


# ------------------------------------------------------------------ scatter estimators

def pd_project(S, eps=1e-3):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return V @ np.diag(np.clip(w, eps, None)) @ V.T


def mad(X):
    m = np.median(X, axis=0)
    s = 1.4826 * np.median(np.abs(X - m), axis=0)
    return np.where(s < 1e-8, 1.0, s)


def cov_sample(X):
    return np.cov(X, rowvar=False)


def cov_skeptic(X):
    """Kendall's tau mapped by sin(pi tau / 2), scaled by MAD standard deviations."""
    R = np.sin(np.pi / 2 * pd.DataFrame(X).corr(method='kendall').values)
    s = mad(X)
    return R * np.outer(s, s)


def cov_skeptic_corr(X):
    """Kendall's tau mapped by sin(pi tau / 2), left on the correlation scale."""
    return np.sin(np.pi / 2 * pd.DataFrame(X).corr(method='kendall').values)


def cov_tyler(X, iters=200, tol=1e-6):
    n, p = X.shape
    Xc = X - np.median(X, axis=0)
    V = np.eye(p)
    for _ in range(iters):
        q = np.clip(np.einsum('ij,jk,ik->i', Xc, np.linalg.inv(V), Xc), 1e-12, None)
        Vn = (p / n) * (Xc / q[:, None]).T @ Xc
        Vn = Vn / np.trace(Vn) * p
        done = np.abs(Vn - V).max() < tol
        V = Vn
        if done:
            break
    d = np.sqrt(np.diag(V))
    s = mad(X)
    return V / np.outer(d, d) * np.outer(s, s)


# ------------------------------------------------------------------ graphical lasso tuning

def glasso_path(S):
    """Precision estimates on the original scale for every lambda (fit on the correlation scale)."""
    S = pd_project(S)
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    out = []
    for lam in LAMBDAS:
        try:
            _, P = graphical_lasso(R, alpha=lam, max_iter=200)
        except Exception:
            P = np.linalg.inv(R + lam * np.eye(len(R)))
        out.append(P / np.outer(d, d))
    return out


def nll(S, P):
    sign, logdet = np.linalg.slogdet(P)
    return np.inf if sign <= 0 else float(np.trace(S @ P) - logdet)


def tune_bic(S, n):
    best, bb = None, np.inf
    for P in glasso_path(S):
        k = int((np.abs(P) > 1e-8).sum() - len(P)) // 2
        b = n * nll(S, P) + k * np.log(n)
        if b < bb:
            bb, best = b, P
    return best


def tune_cv(S_full, X_folds, cov_fn, seed=0, k=3):
    n = X_folds.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    loss = np.zeros(len(LAMBDAS))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        try:
            S_tr, S_va = cov_fn(X_folds[tr]), cov_fn(X_folds[f])
        except Exception:
            continue
        for i, P in enumerate(glasso_path(S_tr)):
            loss[i] += nll(pd_project(S_va), P)
    return glasso_path(S_full)[int(np.argmin(loss))], float(LAMBDAS[int(np.argmin(loss))])


# ------------------------------------------------------------------ data export and R

def stem_h2h(P, Nn, kind, rep):
    return f'h2h_{P}_{Nn}_{kind}_{rep}'


def stem_fault(case, pi, seed):
    return f'fault_{case}_{pi}_{seed}'


def export():
    os.makedirs(DIN, exist_ok=True)
    for (P, Nn) in E.H2H_DESIGNS:
        for kind in ('clean', 'cell', 'row'):
            for rep in E.H2H_REPS:
                Om, X, mask = E.h2h_data(P, Nn, rep, kind)
                np.savetxt(f'{DIN}/{stem_h2h(P, Nn, kind, rep)}_X.csv', X, delimiter=',')
    Z, sds = E.fault_data()
    for (case, pi, s) in E.fault_cases():
        X = Z if case == 'clean' else E.inject(Z, sds, pi, 500 + s)[0]
        np.savetxt(f'{DIN}/{stem_fault(case, pi, s)}_X.csv', X, delimiter=',')
    print('exported')


def load(stem, what):
    f = f'{DOUT}/{stem}_{what}.csv'
    return np.loadtxt(f, delimiter=',') if os.path.exists(f) else None


def estimates(X, stem, n):
    """Yield (method, precision_bic, precision_cv, lambda_cv, detection_scores)."""
    ddcimp, gse = load(stem, 'ddcimp'), load(stem, 'gse')
    mcdS, mcdimp = load(stem, 'mcdS'), load(stem, 'mcdimp')
    specs = [('sample', cov_sample(X), X, cov_sample, None),
             ('SKEPTIC', cov_skeptic(X), X, cov_skeptic, None),
             ('SKEPTIC-corr', cov_skeptic_corr(X), X, cov_skeptic_corr, None),
             ('Tyler', cov_tyler(X), X, cov_tyler, None),
             ('DDC', cov_sample(ddcimp), ddcimp, cov_sample, load(stem, 'ddcres')),
             ('2SGS', gse, ddcimp, cov_sample, None)]
    if mcdS is not None:
        specs.append(('cellMCD', mcdS, mcdimp, cov_sample, load(stem, 'mcdres')))
    for name, S, Xf, fn, det in specs:
        Pb = tune_bic(S, n)
        Pc, lam = tune_cv(S, Xf, fn)
        yield name, Pb, Pc, lam, det


def score():
    recs = []
    for (P, Nn) in E.H2H_DESIGNS:
        iu = np.triu_indices(P, 1)
        for kind in ('clean', 'cell', 'row'):
            for rep in E.H2H_REPS:
                Om, X, mask = E.h2h_data(P, Nn, rep, kind)
                truth = (np.abs(Om[iu]) > 1e-10).astype(int)
                for name, Pb, Pc, lam, det in estimates(X, stem_h2h(P, Nn, kind, rep), Nn):
                    r = dict(job='H2H', P=P, N=Nn, kind=kind, rep=rep, method=name, lam_cv=lam)
                    for tag, Ph in (('bic', Pb), ('cv', Pc)):
                        r[f'relfro_{tag}'] = float(np.linalg.norm(Ph - Om) / np.linalg.norm(Om))
                        r[f'auc_{tag}'] = float(roc_auc_score(truth, np.abs(Ph[iu])))
                    if det is not None and mask.any():
                        r['detect_auc'] = float(roc_auc_score(mask.ravel(), det.ravel()))
                    recs.append(r)
                print('scored', P, Nn, kind, rep, flush=True)
    Z, sds = E.fault_data()
    for (case, pi, s) in E.fault_cases():
        X, mask = (Z, np.zeros(Z.shape, bool)) if case == 'clean' else E.inject(Z, sds, pi, 500 + s)
        for name, Pb, Pc, lam, det in estimates(X, stem_fault(case, pi, s), X.shape[0]):
            d = np.sqrt(np.diag(Pc))
            r = dict(job='FAULT', case=case, pi=pi, seed=s, method=name,
                     pcor=(-Pc / np.outer(d, d)).tolist())
            if det is not None and mask.any():
                r['detect_auc'] = float(roc_auc_score(mask.ravel(), det.ravel()))
            recs.append(r)
    with open(OUT, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    print('wrote', len(recs))


if __name__ == '__main__':
    step = sys.argv[1]
    if step == 'export':
        export()
    elif step == 'r':
        subprocess.run(['Rscript', 'freq_rev2.R'], check=True)
    else:
        score()
