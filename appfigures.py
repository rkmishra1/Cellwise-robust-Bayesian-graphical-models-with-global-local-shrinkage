"""Application figure (revision): (a) posterior divisors on a real return window
(Dirichlet-t, point-in-time universe); (b) realised OOS volatility of GMV portfolios,
with and without turnover costs; (c) per-window OOS volatility."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pd.read_csv('returns_pit.csv', index_col=0, parse_dates=True)
with open('app2_results.json') as f:
    res = json.load(f)
tau = np.load('app2_tau_dirichlet.npy')
oos = np.load('app2_oos_returns.npy')
MODELS = ['gauss', 'classical', 'alternative', 'dirichlet', 'npn', 'ddc']
LAB = ['Gauss', 'class.-$t$', 'alt.-$t$', 'Dirich.-$t$', 'NPN', 'DDC']
COLS = ['#7f7f7f', '#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
NW = res['n_oos']

fig = plt.figure(figsize=(11.5, 3.6), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.15, 1], wspace=0.32)

ax = fig.add_subplot(gs[0])
im = ax.imshow(np.clip(tau, 0, 3), aspect='auto', cmap='viridis_r', vmin=0, vmax=3)
ax.set_xlabel('stock')
ax.set_ylabel('day in window')
ax.set_title(f"(a) $E[\\tau_{{ij}}\\mid X]$, Dirichlet-$t$, returns\n"
             f"point-in-time universe, {tau.shape[1]} stocks", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.9)

ax = fig.add_subplot(gs[1])
vols = [res['vols'][m] * 100 for m in MODELS]
# turnover t (one-way per quarterly rebalance) -> annual cost at 10bp per side:
# t * 2 sides * 4 rebalances * 0.001 = 0.008 t, in percentage points x100 = 0.8 t
turn = [res['turnover'][m] * 4 * 2 * 10 / 100 if res['turnover'][m] else 0
        for m in MODELS]
vadj = [v + t / 100 for v, t in zip(vols, turn)]
x = np.arange(len(MODELS))
ax.bar(x - 0.2, vols, 0.4, color=COLS, alpha=0.9, label='realised vol')
ax.bar(x + 0.2, vadj, 0.4, color=COLS, alpha=0.45, label='+ 10bp turnover cost')
ax.set_xticks(x)
ax.set_xticklabels(LAB, fontsize=8, rotation=20)
ax.set_ylabel('annualised OOS vol (%)')
ax.set_title(f"(b) GMV portfolios, {res['n_windows']} OOS months", fontsize=9)
ax.legend(fontsize=7, frameon=False)
ax.grid(axis='y', linestyle=':', alpha=0.5)

ax = fig.add_subplot(gs[2])
for k, m in enumerate(MODELS):
    per_win = [oos[w * NW:(w + 1) * NW, k].std() * np.sqrt(252) * 100
               for w in range(res['n_windows'])]
    lw = 2.2 if m == 'gauss' else 1.3
    ax.plot(range(1, res['n_windows'] + 1), per_win, 'o-', color=COLS[k], lw=lw,
            ms=4, label=LAB[k])
ax.set_xlabel('window')
ax.set_ylabel('annualised OOS vol (%)')
ax.set_title('(c) by window', fontsize=9)
ax.legend(fontsize=6.5, frameon=False, ncol=2)
ax.grid(linestyle=':', alpha=0.5)

fig.savefig('fig_app.pdf', bbox_inches='tight')
print('fig_app.pdf done')
