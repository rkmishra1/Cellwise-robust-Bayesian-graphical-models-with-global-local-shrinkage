"""Tables (printed) and figures for the second revision, from rev2.jsonl, rev2_freq.jsonl,
grid.jsonl and rev2_diag/.  Usage: python fig_rev2.py"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})
COL = {'gauss': '#7f7f7f', 'classical': '#d62728', 'alternative': '#1f77b4',
       'dirichlet': '#2ca02c'}
LAB = {'gauss': 'Gaussian', 'classical': 'classical-$t$', 'alternative': 'alternative-$t$',
       'dirichlet': 'Dirichlet-$t$'}
MODELS = ['gauss', 'classical', 'alternative', 'dirichlet']
D = 'rev2_diag'


def jl(f):
    return pd.DataFrame([json.loads(l) for l in open(f) if l.strip()]) if os.path.exists(f) else pd.DataFrame()


R2 = jl('rev2.jsonl')
if 'error' in R2:
    print('errors:', R2['error'].notna().sum())
    R2 = R2[R2['error'].isna()]
FQ = jl('rev2_freq.jsonl')
G = jl('grid.jsonl')
G = G[G.job == 'MAIN']


def ms(x):
    return f'{np.mean(x):.3f}\\,\\tiny$\\pm${np.std(x, ddof=1):.3f}'


# ------------------------------------------------------------------ MCMC diagnostics

def split_rhat(ch):
    """ch: chains x draws."""
    m, n = ch.shape
    h = n // 2
    s = np.concatenate([ch[:, :h], ch[:, h:2 * h]])
    W = s.var(axis=1, ddof=1).mean()
    B = h * s.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * W + B / h) / W)) if W > 0 else np.nan


def ess(ch):
    """Bulk ESS from the pooled autocorrelation (Geyer initial positive sequence)."""
    m, n = ch.shape
    x = ch - ch.mean(axis=1, keepdims=True)
    acf = np.zeros(n)
    for c in x:
        f = np.fft.rfft(c, 2 * n)
        acf += np.fft.irfft(f * np.conj(f))[:n] / n
    acf /= m
    var = ch.var(ddof=1) if acf[0] == 0 else acf[0] * n / (n - 1)
    rho = 1 - (ch.var(axis=1, ddof=1).mean() - acf) / var
    rho[0] = 1.0
    tau, t = 1.0, 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += 2 * pair
        t += 2
    return float(m * n / tau)


def fig_mcmc():
    fs = [f'{D}/chain{c}.npz' for c in range(4)]
    if not all(os.path.exists(f) for f in fs):
        return
    C = [np.load(f) for f in fs]
    burn = 500
    series = {r'$\nu$': [c['nu'] for c in C],
              r'$\log\tau_\Omega^2$': [np.log(c['tau_sq']) for c in C],
              r'$\log|\Omega|$': [c['logdet'] for c in C],
              'clusters per obs.': [c['nclust'] for c in C]}
    idx, truth = C[0]['idx'], C[0]['truth']
    for e in range(len(idx)):
        j, k = idx[e]
        series[fr'$\omega_{{{j + 1},{k + 1}}}$ (true {truth[e]:.2f})'] = [c['omega'][:, e] for c in C]
    fig, axs = plt.subplots(4, 2, figsize=(7.2, 7.4))
    rows = []
    for ax, (name, ss) in zip(axs.ravel(), series.items()):
        for i, s in enumerate(ss):
            ax.plot(s, lw=0.4, alpha=0.8)
        ax.axvline(burn, color='k', lw=0.6, ls=':')
        ch = np.array([s[burn:] for s in ss])
        rh, es = split_rhat(ch), ess(ch)
        rows.append((name, rh, es))
        if np.isnan(rh):
            ax.set_title(f'{name}: identical in all chains at {ch.flat[0]:g}', fontsize=8)
        else:
            ax.set_title(f'{name}   $\\hat R$={rh:.2f}, ESS={es:.0f}', fontsize=8)
        ax.set_xlabel('sweep')
    fig.tight_layout()
    fig.savefig('fig_mcmc.pdf')
    plt.close(fig)
    # R-hat and ESS over all off-diagonal entries from chain 0 is not available (draws only kept
    # for chain 0), so report the traced quantities.
    print('\n=== MCMC diagnostics (4 chains, p=40, cellwise, Dirichlet-t) ===')
    for r in rows:
        print(f'{r[0]:40s} Rhat={r[1]:.3f} ESS={r[2]:.0f}')


def fig_nu():
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.8), sharey='row')
    grid_u = np.concatenate([np.arange(2.5, 10.0, 0.5), np.arange(10.0, 61.0, 2.0)])
    for r, model in enumerate(['classical', 'dirichlet']):
        for c, p in enumerate([40, 80, 120]):
            f = f'{D}/nu_{model}_{p}.npz'
            if not os.path.exists(f):
                continue
            z = np.load(f)
            burn = {40: 250, 80: 200, 120: 200}[p]
            ax = axs[r, c]
            bins = np.arange(2.25, 62, 1.0)
            ax.hist(z['nu'][burn:], bins=bins, color='#d62728', alpha=0.6,
                    label='uniform on grid ($\\leq 30$)', density=True)
            ax.hist(z['nu_prior'][burn:], bins=bins, color='#1f77b4', alpha=0.6,
                    label='Ga(2, 0.1) prior', density=True)
            ax.set_title(f'{LAB[model]}, $p={p}$', fontsize=8)
            ax.set_xlabel(r'$\nu$')
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig('fig_nu.pdf')
    plt.close(fig)


def fig_edges():
    fd, fg = f'{D}/chain0.npz', f'{D}/gauss_draws.npz'
    if not (os.path.exists(fd) and os.path.exists(fg)):
        return
    zd, zg = np.load(fd), np.load(fg)
    Om = zd['Om']
    p = Om.shape[0]
    fig = plt.figure(figsize=(7.2, 5.4))
    mats = [('truth', (np.abs(Om) > 1e-10).astype(float))]
    for name, z in (('Gaussian', zg), ('Dirichlet-$t$', zd)):
        dr = z['draws']
        mats.append((name, (np.abs(dr) > 0.05).mean(axis=0)))
    for i, (name, M) in enumerate(mats):
        ax = fig.add_subplot(2, 3, i + 1)
        M = M.copy()
        np.fill_diagonal(M, np.nan)
        im = ax.imshow(M, cmap='Greys', vmin=0, vmax=1)
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes[:3], shrink=0.7)
    ax = fig.add_subplot(2, 1, 2)
    iu = np.triu_indices(p, 1)
    edges = np.where(np.abs(Om[iu]) > 1e-10)[0]
    order = edges[np.argsort(Om[iu][edges])]
    x = np.arange(len(order))
    for off, (name, z, col) in enumerate((('Gaussian', zg, COL['gauss']),
                                          ('Dirichlet-$t$', zd, COL['dirichlet']))):
        dr = z['draws'][:, iu[0], iu[1]][:, order]
        lo, md, hi = np.quantile(dr, [0.025, 0.5, 0.975], axis=0)
        ax.errorbar(x + (off - 0.5) * 0.3, md, yerr=[md - lo, hi - md], fmt='o', ms=2,
                    lw=0.8, color=col, label=name)
    ax.plot(x, Om[iu][order], 'k_', ms=8, label='true value')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('true edges, ordered by true value')
    ax.set_ylabel(r'$\omega_{jk}$: median and 95% interval')
    ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.savefig('fig_edges.pdf')
    plt.close(fig)


# ------------------------------------------------------------------ fixed nu and nu prior

def tables_grid():
    if R2.empty:
        return
    for job in ('FIXNU', 'NUPRIOR'):
        d = R2[R2.job == job]
        if d.empty:
            continue
        print(f'\n=== {job}: relfro and AUC (mean +- sd) ===')
        for kind in ('cell', 'row', 'clean'):
            print(f'-- {kind}')
            for p in sorted(d.p.dropna().unique()):
                cells = []
                for m in MODELS:
                    x = d[(d.kind == kind) & (d.p == p) & (d.model == m)]
                    if len(x):
                        cells.append(f'{m[:4]} {ms(x.relfro)} / {ms(x.auc)} (n={len(x)})')
                print(int(p), ' | '.join(cells))
        c = d[d.kind == 'cell']
        if 'detect_auc' in c:
            print('-- cell detection AUC and cells discarded')
            print(c.groupby(['p', 'model'])[['detect_auc', 'cells_disc', 'weight', 'nu']]
                  .mean().round(3).to_string())
    f = R2[R2.job == 'FIXNU']
    if f.empty:
        return
    from scipy import stats
    print('\n--- paired tests, FIXNU cellwise, Dirichlet vs others')
    for p in sorted(f.p.dropna().unique()):
        for other in ('classical', 'alternative'):
            for col in ('relfro', 'auc'):
                a = f[(f.p == p) & (f.kind == 'cell') & (f.model == 'dirichlet')].sort_values('rep')[col].values
                b = f[(f.p == p) & (f.kind == 'cell') & (f.model == other)].sort_values('rep')[col].values
                if len(a) == len(b) and len(a) > 1:
                    t, pv = stats.ttest_rel(a, b)
                    print(f'p={int(p)} {col} D vs {other}: {a.mean():.3f} vs {b.mean():.3f} t={t:.2f} p={pv:.4f}')
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3))
    for ax, (kind, col, ttl) in zip(axs, [('cell', 'relfro', 'cellwise: rel. Frobenius'),
                                          ('cell', 'auc', 'cellwise: edge AUC'),
                                          ('clean', 'relfro', 'clean: rel. Frobenius'),
                                          ('cell', 'detect_auc', 'cellwise: cell detection AUC')]):
        for m in MODELS:
            x = f[(f.kind == kind) & (f.model == m)]
            if col not in x or x[col].isna().all():
                continue
            g = x.groupby('p')[col].agg(['mean', 'std'])
            ax.errorbar(g.index, g['mean'], yerr=g['std'], marker='o', ms=3, lw=1,
                        color=COL[m], label=LAB[m], capsize=2)
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel('$p$')
        ax.set_xticks([40, 80, 120])
    axs[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig('fig_fixnu.pdf')
    plt.close(fig)


def tables_alpha():
    a = R2[R2.job.isin(['ALPHA', 'ALPHAFIX'])]
    if a.empty:
        return
    from scipy.special import gammaln

    def eocc(p, K, al):
        b, c = al / K, al * (K - 1) / K
        return K * (1 - np.exp(gammaln(al) + gammaln(c + p) - gammaln(c) - gammaln(al + p)))

    print('\n=== alpha prior (K=10, nu=10) ===')
    rows = []
    for (job, p, kind), x in a.groupby(['job', 'p', 'kind']):
        r = dict(job=job, p=int(p), kind=kind, relfro=ms(x.relfro), auc=ms(x.auc),
                 clusters=f'{x.n_clusters.mean():.2f}')
        if job == 'ALPHA':
            am = x.alpha_mean.mean()
            q = np.array(x.alpha_q.tolist())
            r.update(alpha=f'{am:.2f} [{q[:, 0].mean():.2f}, {q[:, 2].mean():.2f}]',
                     prior_clusters_at_alpha=f'{eocc(p, 10, am):.2f}')
        else:
            r.update(prior_clusters_at_alpha=f'{eocc(p, 10, 1.0):.2f}')
        rows.append(r)
    print(pd.DataFrame(rows).to_string())
    x = a[a.job == 'ALPHA']
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for i, ((p, kind), g) in enumerate(x.groupby(['p', 'kind'])):
        q = np.array(g.alpha_q.tolist())
        axs[0].errorbar(np.full(len(q), i) + np.linspace(-0.2, 0.2, len(q)), q[:, 1],
                        yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt='o', ms=3,
                        color=COL['dirichlet'])
    labels = [f'$p={int(p)}$\n{k}' for (p, k), _ in x.groupby(['p', 'kind'])]
    axs[0].set_xticks(range(len(labels)))
    axs[0].set_xticklabels(labels)
    axs[0].set_ylabel(r'posterior $\alpha$ (median, 95%)')
    axs[0].axhline(1.0, color='k', lw=0.5, ls=':')
    g = np.linspace(0.02, 5, 200)
    axs[1].plot(g, np.exp(-g), 'k', lw=1, label='Ga(1,1) prior')
    for (p, kind), gg in x.groupby(['p', 'kind']):
        axs[1].axvline(gg.alpha_mean.mean(), lw=1, ls='-' if kind == 'cell' else '--',
                       color=COL['dirichlet'] if p == 40 else COL['alternative'],
                       label=f'posterior mean, $p={int(p)}$, {kind}')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig('fig_alpha.pdf')
    plt.close(fig)


# ------------------------------------------------------------------ head-to-head

def tables_h2h():
    b = R2[R2.job == 'H2H'] if not R2.empty else pd.DataFrame()
    if b.empty or FQ.empty:
        return
    f = FQ[FQ.job == 'H2H'].copy()
    rows = []
    for (P, Nn), bb in b.groupby(['P', 'N']):
        for m in MODELS:
            r = dict(design=f'{int(P)}/{int(Nn)}', method=LAB[m])
            for kind in ('clean', 'cell', 'row'):
                x = bb[(bb.method == m) & (bb.kind == kind)]
                r[kind] = f'{x.relfro.mean():.3f}/{x.auc.mean():.3f}'
            rows.append(r)
        ff = f[(f.P == P) & (f.N == Nn)]
        for m in ff.method.unique():
            for tag in ('cv', 'bic'):
                r = dict(design=f'{int(P)}/{int(Nn)}', method=f'{m} ({tag.upper()})')
                for kind in ('clean', 'cell', 'row'):
                    x = ff[(ff.method == m) & (ff.kind == kind)]
                    r[kind] = f'{x[f"relfro_{tag}"].mean():.3f}/{x[f"auc_{tag}"].mean():.3f}'
                rows.append(r)
    print('\n=== enlarged head-to-head (relfro/AUC, 10 reps) ===')
    print(pd.DataFrame(rows).to_string())
    print('\nsd of relfro, cellwise:')
    print(b[b.kind == 'cell'].groupby(['P', 'method']).relfro.std().round(3).to_string())
    print(f[f.kind == 'cell'].groupby(['P', 'method']).relfro_cv.std().round(3).to_string())
    print('\ncell detection AUC, cellwise:')
    print(b[b.kind == 'cell'].groupby(['P', 'method']).detect_auc.mean().round(3).to_string())
    print(f[f.kind == 'cell'].groupby(['P', 'method']).detect_auc.mean().round(3).to_string())
    print('\nsecs:', b.groupby(['P', 'method']).secs.mean().round(1).to_dict())

    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex='col')
    meths = [('bayes', m) for m in MODELS] + [('freq', m) for m in
                                              ['sample', 'SKEPTIC', 'Tyler', 'DDC', 'cellMCD', '2SGS']]
    for ci, kind in enumerate(('clean', 'cell', 'row')):
        for ri, (P, Nn) in enumerate([(30, 120), (60, 300)]):
            ax = axs[ri, ci]
            for yi, (fam, m) in enumerate(meths):
                if fam == 'bayes':
                    x = b[(b.P == P) & (b.method == m) & (b.kind == kind)].relfro
                    c = COL[m]
                else:
                    x = f[(f.P == P) & (f.method == m) & (f.kind == kind)].relfro_cv
                    c = 'k'
                if len(x):
                    ax.errorbar(x.mean(), yi, xerr=x.std(), fmt='o', ms=3, color=c, capsize=2)
            ax.set_yticks(range(len(meths)))
            ax.set_yticklabels([LAB.get(m, m) for _, m in meths] if ci == 0 else [])
            ax.invert_yaxis()
            ax.set_title(f'{kind}, $p={P}$, $n={Nn}$', fontsize=8)
            if ri == 1:
                ax.set_xlabel('relative Frobenius error')
    fig.tight_layout()
    fig.savefig('fig_h2h.pdf')
    plt.close(fig)


# ------------------------------------------------------------------ fault injection

def tables_fault():
    b = R2[R2.job == 'FAULT'] if not R2.empty else pd.DataFrame()
    if b.empty:
        return
    f = FQ[FQ.job == 'FAULT'] if not FQ.empty else pd.DataFrame()
    allr = pd.concat([b.assign(fam='bayes'), f.assign(fam='freq')], ignore_index=True)
    rows = []
    for m, g in allr.groupby('method'):
        cl = g[g.case == 'clean']
        if cl.empty:
            continue
        P0 = np.array(cl.pcor.iloc[0])
        iu = np.triu_indices(len(P0), 1)
        top = np.abs(P0[iu]) >= np.quantile(np.abs(P0[iu]), 0.9)
        for pi in (0.02, 0.05):
            gg = g[(g.case == 'fault') & (g.pi == pi)]
            sh, ov = [], []
            for pc in gg.pcor:
                P1 = np.array(pc)
                sh.append(np.linalg.norm(P1[iu] - P0[iu]) / np.linalg.norm(P0[iu]))
                t1 = np.abs(P1[iu]) >= np.quantile(np.abs(P1[iu]), 0.9)
                ov.append((t1 & top).sum() / top.sum())
            rows.append(dict(method=m, pi=pi, shift=f'{np.mean(sh):.3f}\\pm{np.std(sh, ddof=1):.3f}',
                             top10_overlap=f'{np.mean(ov):.2f}',
                             detect=(f'{gg.detect_auc.mean():.3f}' if 'detect_auc' in gg and
                                     gg.detect_auc.notna().any() else '')))
    print('\n=== fault injection (breast cancer, decimal-point faults, 5 seeds) ===')
    print(pd.DataFrame(rows).to_string())
    fm = f'{D}/fault_mask.npy'
    if not os.path.exists(fm):
        return
    mask = np.load(fm)
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.8), gridspec_kw={'width_ratios': [1.3, 1, 1.1]})
    tau = np.load(f'{D}/fault_tau_dirichlet.npy')
    rows_show = slice(0, 60)
    im0 = axs[0].imshow(tau[rows_show].T, aspect='auto', cmap='Greys_r', vmin=0, vmax=1)
    fig.colorbar(im0, ax=axs[0], shrink=0.85, label=r'$\mathbb{E}[\tau_{ij}\mid X]$')
    yy, xx = np.where(mask[rows_show].T)
    axs[0].scatter(xx, yy, s=6, facecolors='none', edgecolors='#d62728', lw=0.6)
    axs[0].set_xlabel('observation (first 60)')
    axs[0].set_ylabel('variable')
    axs[0].set_title('Dirichlet-$t$ divisors; circles = injected faults', fontsize=8)
    for m in ('classical', 'alternative', 'dirichlet'):
        fn = f'{D}/fault_tau_{m}.npy'
        if os.path.exists(fn):
            w = np.broadcast_to(np.load(fn), mask.shape)
            fpr, tpr, _ = roc_curve(mask.ravel(), -w.ravel())
            axs[1].plot(fpr, tpr, color=COL[m], lw=1, label=LAB[m])
    axs[1].plot([0, 1], [0, 1], 'k:', lw=0.5)
    axs[1].set_xlabel('false positive rate')
    axs[1].set_ylabel('true positive rate')
    axs[1].set_title('detection of faulty cells ($\\pi=0.05$)', fontsize=8)
    axs[1].legend(fontsize=6, frameon=False)
    shifts = {}
    for m, g in allr.groupby('method'):
        cl = g[g.case == 'clean']
        gg = g[(g.case == 'fault') & (g.pi == 0.05)]
        if cl.empty or gg.empty:
            continue
        P0 = np.array(cl.pcor.iloc[0])
        iu = np.triu_indices(len(P0), 1)
        shifts[m] = [np.linalg.norm(np.array(pc)[iu] - P0[iu]) / np.linalg.norm(P0[iu]) for pc in gg.pcor]
    names = sorted(shifts, key=lambda n: np.mean(shifts[n]))
    axs[2].barh(range(len(names)), [np.mean(shifts[n]) for n in names],
                xerr=[np.std(shifts[n], ddof=1) for n in names],
                color=[COL.get(n, '0.3') for n in names])
    axs[2].set_yticks(range(len(names)))
    axs[2].set_yticklabels([LAB.get(n, n) for n in names], fontsize=7)
    axs[2].set_xlabel('relative change in\npartial correlations')
    axs[2].set_title('effect of faults on the graph', fontsize=8)
    fig.tight_layout()
    fig.savefig('fig_fault.pdf')
    plt.close(fig)


if __name__ == '__main__':
    fig_mcmc()
    fig_nu()
    fig_edges()
    tables_grid()
    tables_alpha()
    tables_h2h()
    tables_fault()
