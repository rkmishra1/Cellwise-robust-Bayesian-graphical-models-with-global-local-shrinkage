"""A5: demonstrate the uncertainty quantification.  For p=40, clean and cellwise
data, 5 replicates, Gaussian GHS and Dirichlet-t: (i) empirical coverage of central
95% credible intervals for every off-diagonal omega_jk, split by true zero/nonzero;
(ii) reliability of posterior edge probabilities P(|omega_jk| > 0.05) against the true
edge indicator; (iii) AUC of the edge probabilities versus AUC of |posterior mean|.
Saves calibration.json and fig_calibration.pdf.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from grid import gen, contaminate
from dtghs import sample

P, N = 40, 150
DELTA = 0.05


def main():
    iu = np.triu_indices(P, 1)
    out = []
    for kind in ['clean', 'cell']:
        for rep in range(5):
            rng = np.random.default_rng(1000 * rep + P)
            Om, X0 = gen(P, N, rng, seed=rep)
            X, _ = contaminate(X0, rng, kind)
            truth = (np.abs(Om[iu]) > 1e-10)
            for model in ['gauss', 'dirichlet']:
                res = sample(X, model=model, n_iter=700, burn=250, seed=7 + rep,
                             store_omega=True)
                draws = res['omega_draws'][:, iu[0], iu[1]]        # (keep, m)
                lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
                cov = ((Om[iu] >= lo) & (Om[iu] <= hi))
                pedge = (np.abs(draws) > DELTA).mean(axis=0)
                rec = dict(kind=kind, rep=rep, model=model,
                           coverage_all=float(cov.mean()),
                           coverage_zero=float(cov[~truth].mean()),
                           coverage_nonzero=float(cov[truth].mean()),
                           auc_edgerep=float(roc_auc_score(truth.astype(int), pedge)),
                           auc_postmean=float(roc_auc_score(truth.astype(int),
                                                            np.abs(res['Omega'][iu]))),
                           n_edges=int(truth.sum()))
                out.append(rec)
                print(f"{kind:5s} rep{rep} {model:10s} cov={rec['coverage_all']:.3f} "
                      f"(0:{rec['coverage_zero']:.3f}, 1:{rec['coverage_nonzero']:.3f}) "
                      f"AUC(edge)={rec['auc_edgerep']:.3f} "
                      f"AUC(mean)={rec['auc_postmean']:.3f}", flush=True)
    with open('calibration.json', 'w') as f:
        json.dump(out, f, indent=1)

    # figure: coverage bars + reliability-style scatter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), dpi=300)
    ax = axes[0]
    models = ['gauss', 'dirichlet']
    labels = ['Gaussian GHS', 'Dirichlet-$t$']
    width = 0.35
    for k, (m, lab) in enumerate(zip(models, labels)):
        for j, (key, sub) in enumerate([('coverage_zero', 'true zeros'),
                                        ('coverage_nonzero', 'true edges'),
                                        ('coverage_all', 'all entries')]):
            vals = [r[key] for r in out if r['model'] == m and r['kind'] == 'cell']
            ax.bar(j + (k - 0.5) * width, np.mean(vals), width,
                   yerr=np.std(vals), color=['#1f77b4', '#2ca02c'][k],
                   label=lab if j == 0 else None, alpha=0.85)
    ax.axhline(0.95, color='k', ls=':', lw=1)
    ax.text(2.4, 0.955, 'nominal 95%', fontsize=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['true zeros', 'true edges', 'all entries'])
    ax.set_ylabel('95% CI empirical coverage')
    ax.set_title('(a) Credible-interval coverage, cellwise', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    for k, (m, lab) in enumerate(zip(models, labels)):
        pe = [r['auc_edgerep'] for r in out if r['model'] == m]
        pm = [r['auc_postmean'] for r in out if r['model'] == m]
        ax.scatter(np.full(len(pe), k - 0.15), pe, color=['#1f77b4', '#2ca02c'][k],
                   label=f'{lab}: $P(|\\omega_jk|>\\delta)$' if False else lab)
        ax.scatter(np.full(len(pm), k + 0.15), pm, color=['#1f77b4', '#2ca02c'][k],
                   marker='x')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel('edge-ranking AUC')
    ax.set_title('(b) AUC: edge probabilities (o) vs posterior mean (x)', fontsize=9)
    ax.set_ylim(0.9, 1.0)
    ax.grid(axis='y', ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig('fig_calibration.pdf')
    print('saved fig_calibration.pdf', flush=True)


if __name__ == '__main__':
    main()
