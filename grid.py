"""Master grid for the revision: (1) all four models on the main simulation grid with
>=10 replicates at p<=40 and >=5 at p>=80 (Dirichlet-t included for the first time);
(2) K/alpha sensitivity for the Dirichlet-t; (3) contamination magnitude/rate
sensitivity with per-column scales.  Resumable; pre-seeded from headline.jsonl
(identical design and seeds).  Usage: python3 grid.py --shard 0 --nshards 2
"""
import argparse, json, os, sys, time, zlib
import numpy as np
from sklearn.metrics import roc_auc_score
from dtghs import make_precision, sample

OUT = 'grid.jsonl'
N = 150
PS = [20, 40, 80, 120]


def jkey(j):
    return (j['job'], j['p'], j['kind'], j['rep'], j['model'], j.get('K', ''),
            j.get('alpha', ''), j.get('mag', ''), j.get('pi', ''))


def jcrc(j):
    return zlib.crc32(repr(jkey(j)).encode())


def iters(p):
    if p <= 40:
        return (700, 250)
    return (400, 150) if p <= 80 else (250, 100)


def reps_for(p):
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if p <= 40 else [0, 1, 2, 3, 4]


def gen(p, n, rng, seed):
    Om, _ = make_precision(p, density=0.05, signal=0.45, seed=seed)
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(np.linalg.inv(Om)).T
    return Om, X


def contaminate(X, rng, kind, pi=0.02, mag=10.0, per_column=False):
    n, p = X.shape
    Xc = X.copy()
    sd = X.std(axis=0) if per_column else X.std()
    if kind == 'clean':
        mask = np.zeros((n, p), bool)
    elif kind == 'cell':
        mask = rng.random((n, p)) < pi
    elif kind == 'row':
        nrow = max(1, int(round(pi * n)))
        mask = np.zeros((n, p), bool)
        mask[rng.choice(n, nrow, replace=False), :] = True
    else:
        raise ValueError(kind)
    signs = rng.choice([-1.0, 1.0], size=int(mask.sum()))
    if per_column:
        cols = np.broadcast_to(np.arange(p), (n, p))[mask]
        Xc[mask] += mag * sd[cols] * signs
    else:
        Xc[mask] += mag * sd * signs
    return Xc, mask


def done_keys():
    if not os.path.exists(OUT):
        return set()
    ks = set()
    for line in open(OUT):
        try:
            r = json.loads(line)
            ks.add((r['job'], r['p'], r['kind'], r['rep'], r['model'],
                    r.get('K', ''), r.get('alpha', ''), r.get('mag', ''),
                    r.get('pi', '')))
        except Exception:
            pass
    return ks


def one(job, p, kind, rep, model, K=5, alpha=1.0, mag=10.0, pi=0.02,
        per_column=False):
    rng = np.random.default_rng(1000 * rep + p)
    Om, X0 = gen(p, N, rng, seed=rep)
    X, mask = contaminate(X0, rng, kind, pi=pi, mag=mag, per_column=per_column)
    iu = np.triu_indices(p, 1)
    truth = (np.abs(Om[iu]) > 1e-10).astype(int)
    ni, bn = iters(p)

    t0 = time.time()
    R = sample(X, model=model, n_iter=ni, burn=bn, seed=7 + rep, K=K, alpha=alpha)
    Oh = R['Omega']
    rec = dict(job=job, p=p, n=N, kind=kind, rep=rep, model=model,
               relfro=float(np.linalg.norm(Oh - Om, 'fro') / np.linalg.norm(Om, 'fro')),
               auc=float(roc_auc_score(truth, np.abs(Oh[iu]))),
               nu=float(R['nu']), secs=round(time.time() - t0, 1),
               rows_hit=float(mask.any(axis=1).mean()),
               cells_hit=float(mask.mean()),
               K=(K if model == 'dirichlet' else ''),
               alpha=(alpha if model == 'dirichlet' else ''),
               mag=(mag if job == 'MP' else ''), pi=(pi if job == 'MP' else ''))
    if model in ('classical', 'alternative', 'dirichlet'):
        w = R['tau_mean']
        rec['ess_frac'] = float(np.mean(np.minimum(w, 1.0)))
        if model == 'classical':
            rec['rows_killed'] = float(np.mean(w < 0.5))
        else:
            rec['cells_killed'] = float(np.mean(w < 0.5))
            rec['rows_killed'] = float(np.mean((w < 0.5).all(axis=1)))
    if model == 'dirichlet':
        rec['n_clusters'] = float(R['n_clusters'])
    return rec


def build_jobs():
    jobs = []
    for k in ['cell', 'row', 'clean']:
        for p in PS:
            for r in reps_for(p):
                for m in ['gauss', 'classical', 'alternative', 'dirichlet']:
                    jobs.append(dict(job='MAIN', p=p, kind=k, rep=r, model=m))
    for k in ['cell', 'clean']:
        for K in [5, 10, 20]:
            for a in [0.1, 1.0, 10.0]:
                for r in [0, 1, 2]:
                    jobs.append(dict(job='KA', p=40, kind=k, rep=r, model='dirichlet',
                                     K=K, alpha=a))
    for mag in [3.0, 5.0, 10.0]:
        for pi in [0.02, 0.05, 0.10]:
            for r in [0, 1, 2]:
                for m in ['classical', 'alternative', 'dirichlet']:
                    jobs.append(dict(job='MP', p=40, kind='cell', rep=r, model=m,
                                     mag=mag, pi=pi))
    return jobs


def preseed():
    """Copy headline.jsonl rows (same design/seeds) into grid.jsonl as MAIN."""
    if not os.path.exists('headline.jsonl'):
        return 0
    have = done_keys()
    n = 0
    with open(OUT, 'a') as f:
        for line in open('headline.jsonl'):
            r = json.loads(line)
            k = ('MAIN', r['p'], r['kind'], r['rep'], r['model'], '', '', '', '')
            if k in have:
                continue
            r.update(job='MAIN', K='', alpha='', mag='', pi='')
            f.write(json.dumps(r) + '\n')
            have.add(k)
            n += 1
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()

    n = preseed()
    if n:
        print(f"preseeded {n} rows from headline.jsonl", flush=True)
    jobs = build_jobs()
    have = done_keys()
    jobs = [j for j in jobs if jkey(j) not in have]
    if args.nshards > 1:
        jobs = [j for j in jobs if jcrc(j) % args.nshards == args.shard]
    print(f"{len(have)} done, {len(jobs)} jobs for shard {args.shard}", flush=True)
    t0 = time.time()
    for j in jobs:
        if time.time() - t0 > 3600 * 6:
            print("time budget reached", flush=True)
            break
        rec = one(**j, per_column=(j['job'] == 'MP'))
        with open(OUT, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        tag = f"K={rec['K']} a={rec['alpha']} mag={rec['mag']} pi={rec['pi']}"
        print(f"[{rec['job']}] p={rec['p']:3d} {rec['kind']:5s} rep{rec['rep']} "
              f"{rec['model']:12s} relFro={rec['relfro']:.3f} AUC={rec['auc']:.3f} "
              f"nu={rec['nu']:.1f} {tag} {rec['secs']}s", flush=True)
    print("SHARD DONE", flush=True)
