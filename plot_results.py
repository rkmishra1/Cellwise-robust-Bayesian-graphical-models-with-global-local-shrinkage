import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_plots():
    if not os.path.exists('grid.jsonl'):
        print("grid.jsonl not found.")
        return

    rows = []
    with open('grid.jsonl') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('job', 'MAIN') == 'MAIN':
                    rows.append(r)
    
    if not rows:
        print("No data in grid.jsonl")
        return

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Panel (a): Relative Frobenius Error under cellwise contamination vs p
    ax1 = axes[0]
    cell_df = df[df['kind'] == 'cell']
    if not cell_df.empty:
        summary = cell_df.groupby(['p', 'model'])['relfro'].mean().unstack()
        sd = cell_df.groupby(['p', 'model'])['relfro'].std().unstack()
        for model, color, fmt in [('gauss', '#7f7f7f', 's--'), ('classical', '#d62728', 'o-'),
                                  ('alternative', '#1f77b4', '^-'), ('dirichlet', '#2ca02c', 'D-')]:
            if model in summary.columns:
                ax1.plot(summary.index, summary[model], fmt, label=model, linewidth=2, markersize=7 if model != 'dirichlet' else 5, color=color)
                ax1.fill_between(summary.index, summary[model] - sd[model],
                                 summary[model] + sd[model], color=color, alpha=0.12)
        ax1.set_xlabel('Dimension p')
        ax1.set_ylabel('Relative Frobenius Error')
        ax1.set_title('(a) Cellwise contamination (pi=0.02), mean +- sd')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()
        
    # Panel (b): Fraction of observations down-weighted vs p
    ax2 = axes[1]
    if not cell_df.empty and 'rows_killed' in cell_df.columns:
        summary_killed = cell_df.groupby(['p', 'model'])['rows_killed'].mean().unstack()
        ps = np.linspace(15, 125, 100)
        pi = 0.02
        prop_curve = 1.0 - (1.0 - pi)**ps
        ax2.plot(ps, prop_curve, 'k:', label='1 - (1-pi)^p (propagation)', linewidth=1.5)
        
        for model, color, fmt in [('classical', '#d62728', 'o-'), ('alternative', '#1f77b4', '^-')]:
            if model in summary_killed.columns:
                ax2.plot(summary_killed.index, summary_killed[model], fmt, label=model, linewidth=2, markersize=7, color=color)
        ax2.set_xlabel('Dimension p')
        ax2.set_ylabel('Fraction of rows down-weighted (<0.5)')
        ax2.set_title('(b) Down-weighting mechanism')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        
    # Panel (c): All contamination regimes at p=40
    ax3 = axes[2]
    p40_df = df[df['p'] == 40]
    if not p40_df.empty:
        summary_p40 = p40_df.groupby(['kind', 'model'])['relfro'].mean().unstack()
        kinds = ['clean', 'cell', 'row']
        x = np.arange(len(kinds))
        width = 0.2
        models = ['gauss', 'classical', 'alternative', 'dirichlet']
        colors = ['#7f7f7f', '#d62728', '#1f77b4', '#2ca02c']

        for i, (m, col) in enumerate(zip(models, colors)):
            if m in summary_p40.columns:
                vals = [summary_p40.loc[k, m] if k in summary_p40.index else 0 for k in kinds]
                ax3.bar(x + (i - 1.5) * width, vals, width, label=m, color=col, alpha=0.85)
                
        ax3.set_xticks(x)
        ax3.set_xticklabels(kinds)
        ax3.set_ylabel('Relative Frobenius Error')
        ax3.set_title('(c) Performance at p=40 across regimes')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend()
        
    plt.tight_layout()
    plt.savefig('fig_crossover.png')
    print("Updated fig_crossover.png successfully.")

if __name__ == '__main__':
    generate_plots()
