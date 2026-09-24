"""
Divisor-mixture graphical horseshoe samplers.

Models
------
'gauss'        x_i ~ N(mu, Omega^{-1})                              (Li, Craig & Bhadra 2019)
'classical'    x_i = mu + tau_i^{-1/2} z_i,  tau_i ~ Ga(nu/2, nu/2)  (multivariate t)
'alternative'  x_i = mu + D_i^{-1/2} z_i,   D_i = diag(tau_i1..tau_ip),
               tau_ij ~ Ga(nu/2, nu/2) independently                (Finegold & Drton alternative t)

In every case z_i ~ N(0, Omega^{-1}) and Omega carries the graphical horseshoe prior
    omega_jk ~ N(0, lambda_jk^2 tau_O^2),  lambda_jk, tau_O ~ C+(0,1),  omega_jj propto 1.

Conditional on the divisors, the whitened data y_i = D_i^{1/2}(x_i - mu) is exactly Gaussian,
so the Omega block is the standard GHS column-wise update applied to S = sum_i y_i y_i'.
"""

import numpy as np
from scipy.special import gammaln, ndtr, ndtri

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _rtruncnorm_pos(rng, m, s):
    """Draw from N(m, s^2) truncated to (0, inf), vectorised, via inverse CDF."""
    lo = ndtr(-m / s)                      # Phi((0-m)/s)
    u = lo + (1.0 - lo) * rng.random(m.shape)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return np.maximum(m + s * ndtri(u), 1e-10)


def _ltruncnorm_pos(x, m, s):
    """log density of N(m, s^2) truncated to (0, inf)."""
    z = (x - m) / s
    lo = ndtr(-m / s)
    return -0.5 * z ** 2 - _LOG_SQRT_2PI - np.log(s) - np.log(np.maximum(1.0 - lo, 1e-300))


# ----------------------------------------------------------------------------- utilities

def rinvgamma(rng, shape, rate):
    """InvGamma(shape, rate): if X ~ Ga(shape, rate) then 1/X ~ InvGamma(shape, rate)."""
    return 1.0 / rng.gamma(shape, 1.0 / rate)


def make_precision(p, density=0.05, signal=0.4, seed=0):
    """Random sparse positive-definite precision matrix."""
    rng = np.random.default_rng(seed)
    A = np.zeros((p, p))
    iu = np.triu_indices(p, 1)
    m = len(iu[0])
    on = rng.random(m) < density
    vals = signal * rng.choice([-1.0, 1.0], size=m)
    A[iu] = np.where(on, vals, 0.0)
    A = A + A.T
    ev = np.linalg.eigvalsh(A).min()
    A += (abs(ev) + 0.2) * np.eye(p)
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)          # unit partial-variance scaling
    return A, (np.abs(A) > 1e-10) & ~np.eye(p, dtype=bool)


# ----------------------------------------------------------------- graphical horseshoe block

def ghs_sweep(S, n, Omega, Lam, nu_l, tau_sq, xi, rng, Sigma=None):
    """One column-wise block Gibbs sweep for the graphical horseshoe.

    S     : p x p scatter matrix of the (whitened) data
    n     : effective sample size
    Omega : current precision, updated in place
    Lam   : p x p matrix of local scales lambda_jk^2 (symmetric, diagonal unused)
    nu_l  : p x p auxiliary variables for the half-Cauchy on lambda
    Sigma : Omega^{-1}, maintained in place.

    NOTE ON COMPLEXITY.  Maintaining Sigma removes the explicit inversion of Omega_11
    at each column (now O(p^2) via the block identity).  It does NOT make the sweep
    O(p^3): drawing beta still requires inverting/factorising the (p-1)x(p-1) matrix
    Cinv, which is O(p^3) per column and hence O(p^4) per sweep.  That is the cost of
    the standard Li-Craig-Bhadra sampler.  Reaching O(p^3) per iteration requires the
    reverse telescoping block decomposition of Gao et al. (2025), which is NOT
    implemented here.  Empirically this caps the present prototype near p ~ 100.

    Uses the two standard block-inverse identities.  With j moved last,
        Omega_11^{-1} = Sigma_11 - sigma_12 sigma_12' / sigma_22,
    and after drawing (beta, gamma) the refreshed inverse is
        Sigma_11 = Omega_11^{-1} + (Omega_11^{-1}beta)(Omega_11^{-1}beta)'/gamma,
        Sigma_12 = -(Omega_11^{-1}beta)/gamma,   Sigma_22 = 1/gamma.
    """
    p = Omega.shape[0]
    if Sigma is None:
        Sigma = np.linalg.inv(Omega)
    for j in range(p):
        idx = np.arange(p) != j
        s12 = S[idx, j]
        s22 = S[j, j]
        lam12 = Lam[idx, j]

        # Omega_11^{-1} from the maintained inverse -- O(p^2)
        sg12 = Sigma[idx, j]
        sg22 = Sigma[j, j]
        invO11 = Sigma[np.ix_(idx, idx)] - np.outer(sg12, sg12) / sg22
        invO11 = 0.5 * (invO11 + invO11.T)

        Cinv = s22 * invO11 + np.diag(1.0 / (lam12 * tau_sq))
        C = np.linalg.inv(Cinv)
        C = 0.5 * (C + C.T)
        # jitter for numerical symmetry/PD
        L = np.linalg.cholesky(C + 1e-10 * np.eye(p - 1))
        mean = -C @ s12
        beta = mean + L @ rng.standard_normal(p - 1)

        gam = rng.gamma(n / 2.0 + 1.0, 2.0 / s22)
        Ob = invO11 @ beta
        Omega[idx, j] = beta
        Omega[j, idx] = beta
        Omega[j, j] = gam + beta @ Ob

        # refresh Sigma = Omega^{-1} in place -- O(p^2)
        Sigma[np.ix_(idx, idx)] = invO11 + np.outer(Ob, Ob) / gam
        Sigma[idx, j] = -Ob / gam
        Sigma[j, idx] = -Ob / gam
        Sigma[j, j] = 1.0 / gam

        # local shrinkage for this column
        om12 = beta
        rate_l = 1.0 / nu_l[idx, j] + om12 ** 2 / (2.0 * tau_sq)
        new_lam = rinvgamma(rng, 1.0, rate_l)
        Lam[idx, j] = new_lam
        Lam[j, idx] = new_lam
        new_nu = rinvgamma(rng, 1.0, 1.0 + 1.0 / new_lam)
        nu_l[idx, j] = new_nu
        nu_l[j, idx] = new_nu

    # global shrinkage
    iu = np.triu_indices(p, 1)
    q = len(iu[0])
    rate_t = 1.0 / xi + np.sum(Omega[iu] ** 2 / Lam[iu]) / 2.0
    tau_sq = rinvgamma(rng, (q + 1.0) / 2.0, rate_t)
    xi = rinvgamma(rng, 1.0, 1.0 + 1.0 / tau_sq)
    return tau_sq, xi, Sigma


# ------------------------------------------------------------------------- divisor blocks

def update_classical(Xc, Omega, nu, rng):
    """tau_i | . ~ Ga((nu+p)/2, (nu + x_i' Omega x_i)/2).  Returns n-vector."""
    n, p = Xc.shape
    quad = np.einsum('ij,jk,ik->i', Xc, Omega, Xc)
    return rng.gamma((nu + p) / 2.0, 2.0 / (nu + quad))


def update_alternative(Xc, Omega, nu, S_cur, rng):
    """Coordinate-wise divisors via independence Metropolis on s_ij = sqrt(tau_ij).

    Target (per observation i, coordinate j, given the others):
        log p(s) = nu*log(s) - 0.5*(A_jj + nu)*s^2 - b_j*s,   s > 0,
        A = diag(x_i) Omega diag(x_i),  b_j = sum_{k!=j} A_jk s_k.

    Strictly log-concave; mode available in closed form, so a truncated-normal
    Laplace proposal gives a near-perfect independence sampler.
    """
    n, p = Xc.shape
    S = S_cur
    acc = 0
    for j in range(p):
        # A_jj and b_j vectorised over observations
        xj = Xc[:, j]
        A_jj = Omega[j, j] * xj ** 2
        # b_j = x_ij * sum_{k != j} Omega_jk x_ik s_ik
        M = Xc * Omega[j, :][None, :]             # n x p : x_ik * Omega_jk
        b = xj * ((M * S).sum(axis=1) - Omega[j, j] * xj * S[:, j])

        a = A_jj + nu
        # mode: a s^2 + b s - nu = 0
        star = (-b + np.sqrt(b ** 2 + 4.0 * a * nu)) / (2.0 * a)
        star = np.maximum(star, 1e-8)
        prec = nu / star ** 2 + a
        sd = 1.0 / np.sqrt(prec)

        prop = _rtruncnorm_pos(rng, star, sd)
        cur = np.maximum(S[:, j], 1e-10)

        def logt(s):
            return nu * np.log(s) - 0.5 * a * s ** 2 - b * s

        logr = (logt(prop) - _ltruncnorm_pos(prop, star, sd)) \
             - (logt(cur) - _ltruncnorm_pos(cur, star, sd))
        take = np.log(rng.random(n)) < logr
        S[take, j] = prop[take]
        acc += take.sum()
    return S, acc / (n * p)


def update_dirichlet(Xc, Omega, nu, S_cur, Z, U, alpha, rng, K):
    """Dirichlet-t divisors via the finite-K approximation to the DP.

    Within each observation the p coordinates are allocated to K clusters,
        z_ij ~ Cat(pi_i),  pi_i ~ Dir(alpha/K, ..., alpha/K),
        theta_ik ~ Ga(nu/2, nu/2),   tau_ij = theta_{i, z_ij},
    so alpha -> 0 recovers the classical t (all coordinates share one divisor) and
    alpha -> infinity recovers the alternative t (every coordinate its own).

    Writing u_ik = sqrt(theta_ik) and s_i = M_i u_i with M_i the 0/1 allocation
    matrix, the conditional of u_i is
        log p(u_i) = sum_k (nu - 1 + n_ik) log u_ik - (nu/2)||u_i||^2
                     - 0.5 u_i' (M_i' A_i M_i) u_i,
    which is strictly log-concave by the same Hessian argument as Proposition 1,
    with the closed-form mode of Proposition 2 and nu replaced by nu - 1 + n_ik.
    """
    n, p = Xc.shape
    S = S_cur

    # ---- 1. reallocate coordinates, one j at a time (vectorised over observations)
    for j in range(p):
        xj = Xc[:, j]
        A_jj = Omega[j, j] * xj ** 2                       # (n,)
        M = Xc * Omega[j, :][None, :]
        b = xj * ((M * S).sum(axis=1) - Omega[j, j] * xj * S[:, j])   # (n,)

        counts = np.zeros((n, K))
        for k in range(K):
            counts[:, k] = (Z == k).sum(axis=1)
        counts[np.arange(n), Z[:, j]] -= 1                 # remove j itself

        cand = U                                            # (n, K) candidate values
        loglik = (np.log(np.maximum(cand, 1e-300))
                  - 0.5 * A_jj[:, None] * cand ** 2
                  - b[:, None] * cand)
        logw = np.log(counts + alpha / K) + loglik
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        u = rng.random(n)
        Z[:, j] = (w.cumsum(axis=1) < u[:, None]).sum(axis=1).clip(0, K - 1)
        S[:, j] = U[np.arange(n), Z[:, j]]

    # ---- 2. refresh cluster values u_ik
    acc = 0
    for k in range(K):
        mask = (Z == k)                                     # (n, p)
        nk = mask.sum(axis=1)
        act = nk > 0
        if not act.any():
            U[~act, k] = np.sqrt(rng.gamma(nu / 2.0, 2.0 / nu, size=(~act).sum()))
            continue
        # a_ik = sum_{j,l in cluster} A_jl ; b_ik = sum_{j in k} sum_{l not in k} A_jl s_l
        Xm = Xc * mask                                      # zero outside cluster
        a = np.einsum('ij,jl,il->i', Xm, Omega, Xm) + nu
        Xo = Xc * (~mask)
        b = np.einsum('ij,jl,il->i', Xm, Omega, Xo * S)
        m = nu - 1.0 + nk
        star = (-b + np.sqrt(b ** 2 + 4.0 * a * np.maximum(m, 1e-8))) / (2.0 * a)
        star = np.maximum(star, 1e-8)
        prec = np.maximum(m, 1e-8) / star ** 2 + a
        sd = 1.0 / np.sqrt(prec)
        prop = _rtruncnorm_pos(rng, star, sd)
        cur = np.maximum(U[:, k], 1e-10)

        def lt(v):
            return m * np.log(v) - 0.5 * a * v ** 2 - b * v

        logr = (lt(prop) - _ltruncnorm_pos(prop, star, sd)) \
             - (lt(cur) - _ltruncnorm_pos(cur, star, sd))
        take = act & (np.log(rng.random(n)) < logr)
        U[take, k] = prop[take]
        acc += take.sum()

    for j in range(p):
        S[:, j] = U[np.arange(n), Z[:, j]]
    return S, Z, U, acc / (n * K)


def update_nu(T, nu_grid, rng, per_obs_dim=1, logprior=None):
    """Griddy Gibbs for the degrees of freedom given divisors T (flattened).
    logprior: optional log prior weights on nu_grid (default uniform on the grid)."""
    t = np.clip(T.ravel(), 1e-12, None)
    m = t.size
    a = nu_grid / 2.0
    ll = m * (a * np.log(a) - gammaln(a)) + (a - 1.0) * np.sum(np.log(t)) - a * np.sum(t)
    if logprior is not None:
        ll = ll + logprior
    ll -= ll.max()
    w = np.exp(ll)
    w /= w.sum()
    return rng.choice(nu_grid, p=w)


ALPHA_GRID = np.exp(np.linspace(np.log(0.02), np.log(50.0), 160))


def update_alpha(Z, K, p, rng, a0=1.0, b0=1.0):
    """Griddy Gibbs for the Dirichlet concentration alpha ~ Ga(a0, b0) under the finite-K
    model with pi_i ~ Dir(alpha/K,...) integrated out.  Grid is uniform in log(alpha),
    so the Jacobian alpha is included in the prior weight."""
    n = Z.shape[0]
    counts = np.stack([(Z == k).sum(axis=1) for k in range(K)], axis=1)   # n x K
    g = ALPHA_GRID
    ll = n * (gammaln(g) - gammaln(g + p))
    ll = ll + (gammaln(counts[:, :, None] + g[None, None, :] / K)
               - gammaln(g / K)[None, None, :]).sum(axis=(0, 1))
    ll = ll + a0 * np.log(g) - b0 * g          # Ga(a0,b0) density times Jacobian g
    ll -= ll.max()
    w = np.exp(ll)
    w /= w.sum()
    return rng.choice(g, p=w)


# ------------------------------------------------------------------------------ main sampler

def sample(X, model='gauss', n_iter=3000, burn=1000, nu=None, seed=1, thin=2,
           nu_grid=None, verbose=False, alpha=1.0, K=5, store_omega=False,
           nu_logprior=None, alpha_prior=None, trace=False, trace_idx=None):
    """Run the sampler. Returns dict with posterior mean Omega and diagnostics."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)

    Omega = np.eye(p)
    Sigma = np.eye(p)
    Lam = np.ones((p, p))
    nu_l = np.ones((p, p))
    tau_sq, xi = 1.0, 1.0

    sample_nu = nu is None
    if sample_nu:
        nu = 5.0
        if nu_grid is None:
            nu_grid = np.concatenate([np.arange(2.5, 10.0, 0.5), np.arange(10.0, 31.0, 2.0)])
        if callable(nu_logprior):
            nu_logprior = nu_logprior(nu_grid)

    S_div = np.ones((n, p))    # s_ij = sqrt(tau_ij) for 'alternative'/'dirichlet'
    tau_c = np.ones(n)         # tau_i for 'classical'
    Zc = rng.integers(0, K, size=(n, p))      # allocations for 'dirichlet'
    Uc = np.ones((n, K))                      # cluster values sqrt(theta_ik)

    Om_sum = np.zeros((p, p))
    keep = 0
    tau_sum = np.zeros((n, p)) if model in ('classical', 'alternative', 'dirichlet') else None
    nclust_sum = 0.0          # posterior-mean occupied clusters per observation (dirichlet)
    omega_draws = []          # kept Omega draws, only when store_omega=True
    acc_track = []
    nu_track = []
    tr = {'tau_sq': [], 'alpha': [], 'nclust': [], 'omega': [], 'logdet': []}
    if trace and trace_idx is None:
        trace_idx = []

    for it in range(n_iter):
        # --- divisor block -> whitened data
        if model == 'gauss':
            Y = Xc
        elif model == 'classical':
            tau_c = update_classical(Xc, Omega, nu, rng)
            Y = Xc * np.sqrt(tau_c)[:, None]
            if sample_nu:
                nu = update_nu(tau_c, nu_grid, rng, logprior=nu_logprior)
        elif model == 'alternative':
            S_div, ar = update_alternative(Xc, Omega, nu, S_div, rng)
            acc_track.append(ar)
            Y = Xc * S_div
            if sample_nu:
                nu = update_nu(S_div ** 2, nu_grid, rng, logprior=nu_logprior)
        elif model == 'dirichlet':
            S_div, Zc, Uc, ar = update_dirichlet(Xc, Omega, nu, S_div, Zc, Uc,
                                                 alpha, rng, K)
            acc_track.append(ar)
            Y = Xc * S_div
            if sample_nu:
                nu = update_nu(Uc ** 2, nu_grid, rng, logprior=nu_logprior)
            if alpha_prior is not None:
                alpha = update_alpha(Zc, K, p, rng, *alpha_prior)
        else:
            raise ValueError(model)

        nu_track.append(nu)
        S = Y.T @ Y
        tau_sq, xi, Sigma = ghs_sweep(S, n, Omega, Lam, nu_l, tau_sq, xi, rng, Sigma)
        if (it % 200) == 0:      # periodic refresh to control drift
            Sigma = np.linalg.inv(Omega)
        if trace:
            tr['tau_sq'].append(tau_sq)
            tr['alpha'].append(alpha)
            tr['omega'].append([Omega[j, k] for (j, k) in trace_idx])
            tr['logdet'].append(np.linalg.slogdet(Omega)[1])
            if model == 'dirichlet':
                tr['nclust'].append(np.mean([len(np.unique(z)) for z in Zc]))

        if it >= burn and (it - burn) % thin == 0:
            Om_sum += Omega
            keep += 1
            if model in ('classical', 'alternative', 'dirichlet'):
                kept = S_div ** 2 if model != 'classical' else tau_c[:, None]
                tau_sum += kept
            if model == 'dirichlet':
                nclust_sum += np.mean([len(np.unique(z)) for z in Zc])
            if store_omega:
                omega_draws.append(Omega.copy())
        if verbose and (it + 1) % 500 == 0:
            print(f"  [{model}] iter {it+1}/{n_iter}  nu={nu:.1f}", flush=True)

    out = {'Omega': Om_sum / keep, 'nu': np.mean(nu_track[burn:]),
           'nu_trace': np.asarray(nu_track),
           'acc': np.mean(acc_track) if acc_track else None}
    if model in ('alternative', 'dirichlet'):
        out['divisors'] = S_div ** 2
    if model == 'dirichlet':
        # posterior-mean number of occupied divisor clusters per observation:
        # ~1 means the classical t, ~K means the alternative t at the truncation cap
        out['n_clusters'] = float(nclust_sum / keep) if keep else float(
            np.mean([len(np.unique(z)) for z in Zc]))
    if model == 'classical':
        out['divisors'] = tau_c
    if model in ('classical', 'alternative', 'dirichlet'):
        out['tau_mean'] = tau_sum / keep
    if store_omega:
        out['omega_draws'] = np.asarray(omega_draws)
    if trace:
        out['trace'] = {k: np.asarray(v) for k, v in tr.items()}
    if alpha_prior is not None:
        out['alpha_mean'] = float(np.mean(np.asarray(tr['alpha'])[burn:])) if trace else float(alpha)
    return out
