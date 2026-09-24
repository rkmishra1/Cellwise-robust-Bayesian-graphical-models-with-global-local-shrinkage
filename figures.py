"""Standalone figures: (1) Theorem 1 divisor rates; (2) divisor heatmaps by geometry;
(3) head-to-head bars; (4) application panels."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dtghs import make_precision, sample

PI, MAG = 0.02, 10.0


# ---------------------------------------------------------------- Theorem 1
def conditional_maximiser(A, nu, s0=None, iters=500, tol=1e-12):
    """Coordinate ascent on eq:divisor; each update is the exact conditional mode (Prop 2)."""
    s = np.ones(A.shape[0]) if s0 is None else s0.copy()
    for _ in range(iters):
        s_prev = s.copy()
        for j in range(A.shape[0]):
            a_j = A[j, j] + nu
            b_j = A[j] @ s - A[j, j] * s[j]
            s[j] = (-b_j + np.sqrt(b_j**2 + 4 * a_j * nu)) / (2 * a_j)
        if np.max(np.abs(s - s_prev)) < tol:
            break
    return s


def fig_theorem():
    rng = np.random.default_rng(3)
    p, nu = 8, 4.0
    Om, _ = make_precision(p, density=0.05, signal=0.45, seed=5)
    x = rng.standard_normal(p)
    x[x == 0] = 0.7                               # clean coordinates fixed and nonzero
    j = 3                                         # contaminated coordinate
    eps_grid = np.logspace(1, 4, 10)
    tau_j, tau_lo, tau_hi = [], [], []
    for eps in eps_grid:
        xc = x.copy()
        xc[j] = eps
        A = np.outer(xc, xc) * Om
        s = conditional_maximiser(A, nu)
        tau = s**2
        tau_j.append(tau[j])
        clean = np.delete(tau, j)
        tau_lo.append(clean.min())
        tau_hi.append(clean.max())
    tau_j, tau_lo, tau_hi = map(np.array, (tau_j, tau_lo, tau_hi))

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
    ax.loglog(eps_grid, tau_j, 'o-', color='#d62728', lw=2, label=r'$\tau^\star_{j}$, contaminated')
    ax.loglog(eps_grid, eps_grid**-2, 'k:', lw=1.2, label=r'$\propto\varepsilon^{-2}$')
    ax.fill_between(eps_grid, tau_lo, tau_hi, color='#1f77b4', alpha=0.25,
                    label=r'clean $\tau^\star_{k}$ range')
    ax.set_xlabel(r'contamination magnitude $\varepsilon$ (data s.d.)')
    ax.set_ylabel(r'conditional mode of divisor')
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig('fig_theorem.pdf')
    print('fig_theorem.pdf: final tau ratio per decade =',
          np.round(tau_j[:-1] / tau_j[1:], 2))


# ---------------------------------------------------------------- divisors by geometry
def contaminate(X, rng, kind):
    n, p = X.shape
    Xc, sd = X.copy(), X.std()
    if kind == 'clean':
        mask = np.zeros((n, p), bool)
    elif kind == 'cell':
        mask = rng.random((n, p)) < PI
    elif kind == 'row':
        nrow = max(1, int(round(PI * n)))
        mask = np.zeros((n, p), bool)
        mask[rng.choice(n, nrow, replace=False), :] = True
    Xc[mask] += MAG * sd * rng.choice([-1.0, 1.0], size=int(mask.sum()))
    return Xc, mask


def fig_divisors():
    p, n = 40, 100
    rng = np.random.default_rng(9)
    Om, X0 = make_precision(p, density=0.05, signal=0.45, seed=9)
    X0 = rng.standard_normal((n, p)) @ np.linalg.cholesky(np.linalg.inv(Om)).T

    panels = [('classical', 'cell'), ('alternative', 'cell'), ('dirichlet', 'cell'),
              ('classical', 'row'), ('alternative', 'row'), ('dirichlet', 'row')]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), dpi=300)
    for ax, (model, kind) in zip(axes.flat, panels):
        X, mask = contaminate(X0, np.random.default_rng(100 + len(kind)), kind)
        res = sample(X, model=model, n_iter=400, burn=150, seed=13)
        tau = res['tau_mean']
        im = ax.imshow(np.clip(tau, 0, 2.5), aspect='auto', cmap='viridis_r',
                       vmin=0, vmax=2.5)
        ax.set_title(f"{model}-{kind}  (low $\\tau$ = distrusted)", fontsize=9)
        if kind == 'cell':
            ax.text(0.02, 0.02, f"cells contaminated: {mask.mean()*100:.1f}%",
                    transform=ax.transAxes, fontsize=7, color='w')
        else:
            ax.text(0.02, 0.02, f"rows contaminated: {mask.mean()*100:.1f}%",
                    transform=ax.transAxes, fontsize=7, color='w')
    fig.colorbar(im, ax=axes, shrink=0.85, label=r'posterior mean divisor $E[\tau_{ij}\mid X]$')
    fig.savefig('fig_divisors.pdf', bbox_inches='tight')
    print('fig_divisors.pdf done')


if __name__ == '__main__':
    fig_theorem()
    fig_divisors()
