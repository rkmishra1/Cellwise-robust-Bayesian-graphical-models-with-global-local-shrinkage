"""Second-revision experiments, run in parallel and appended to rev2.jsonl.

Jobs
  FIXNU    main grid (p = 40, 80, 120) with nu fixed at 10, all four models
  NUPRIOR  p = 80, 120 with nu sampled under a Ga(2, 0.1) prior on a grid up to 60
  ALPHA    Dirichlet-t with alpha ~ Ga(1, 1), K = 10, nu = 10 (p = 40, 80)
  ALPHAFIX the same runs with alpha fixed at 1, for comparison
  H2H      Bayesian models on the enlarged head-to-head designs (30/120 and 60/300)
  FAULT    decimal-point faults injected into the Wisconsin breast cancer data
  DIAG     multi-chain runs and stored draws for the MCMC diagnostic figures

Usage: OMP_NUM_THREADS=1 python exp_rev2.py [--jobs FIXNU,NUPRIOR,...] [--workers 7]
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
from sklearn.metrics import roc_auc_score
from dtghs import sample
from grid import gen, contaminate
import headline

OUT = 'rev2.jsonl'
DIAG_DIR = 'rev2_diag'
N = 150
NU_WIDE = np.concatenate([np.arange(2.5, 10.0, 0.5), np.arange(10.0, 61.0, 2.0)])


def ga_2_01(g):
    return np.log(g) - 0.1 * g          # Ga(2, 0.1) log density up to a constant


def iters(p):
    return {20: (700, 250), 40: (700, 250), 80: (600, 200), 120: (500, 200)}.get(p, (700, 250))


# ------------------------------------------------------------------ simulation grid

def grid_data(p, kind, rep):
    """Identical data to grid.py (same seeds)."""
    rng = np.random.default_rng(1000 * rep + p)
    Om, X0 = gen(p, N, rng, seed=rep)
    X, mask = contaminate(X0, rng, kind)
    return Om, X, mask


def score(Om, R, mask, model):
    p = Om.shape[0]
    iu = np.triu_indices(p, 1)
    truth = (np.abs(Om[iu]) > 1e-10).astype(int)
    Oh = R['Omega']
    rec = dict(relfro=float(np.linalg.norm(Oh - Om, 'fro') / np.linalg.norm(Om, 'fro')),
               auc=float(roc_auc_score(truth, np.abs(Oh[iu]))), nu=float(R['nu']))
    if model != 'gauss':
        w = R['tau_mean']
        w = np.broadcast_to(w, mask.shape)
        rec['weight'] = float(np.mean(np.minimum(w, 1.0)))
        rec['cells_disc'] = float(np.mean(w < 0.5))
        rec['rows_disc'] = float(np.mean((w < 0.5).all(axis=1)))
        if 0 < mask.sum() < mask.size:
            rec['detect_auc'] = float(roc_auc_score(mask.ravel(), -w.ravel()))
    if model == 'dirichlet':
        rec['n_clusters'] = float(R['n_clusters'])
    return rec


def run_grid(job, p, kind, rep, model):
    Om, X, mask = grid_data(p, kind, rep)
    ni, bn = iters(p)
    kw = dict(n_iter=ni, burn=bn, seed=7 + rep)
    if job == 'FIXNU':
        kw['nu'] = 10.0
    elif job == 'NUPRIOR':
        kw.update(nu_grid=NU_WIDE, nu_logprior=ga_2_01)
    elif job in ('ALPHA', 'ALPHAFIX'):
        kw.update(nu=10.0, K=10, alpha=1.0, trace=True)
        if job == 'ALPHA':
            kw['alpha_prior'] = (1.0, 1.0)
    t0 = time.time()
    R = sample(X, model=model, **kw)
    rec = dict(job=job, p=p, kind=kind, rep=rep, model=model,
               secs=round(time.time() - t0, 1), **score(Om, R, mask, model))
    if job == 'ALPHA':
        rec['alpha_mean'] = R['alpha_mean']
        a = R['trace']['alpha'][bn:]
        rec['alpha_q'] = [float(np.quantile(a, q)) for q in (0.025, 0.5, 0.975)]
    return rec


# ------------------------------------------------------------------ enlarged head-to-head

H2H_DESIGNS = [(30, 120), (60, 300)]
H2H_REPS = list(range(10))


def h2h_data(P, Nn, rep, kind):
    """Same construction as headtohead.data, generalised to (P, Nn)."""
    rng = np.random.default_rng(rep + 97 * P)
    Om, X0 = headline.gen(P, Nn, rng, seed=rep)
    X, mask = headline.contaminate(X0, np.random.default_rng(100 + rep + 97 * P), kind)
    return Om, X, mask


def run_h2h(P, Nn, kind, rep, model):
    Om, X, mask = h2h_data(P, Nn, rep, kind)
    t0 = time.time()
    R = sample(X, model=model, n_iter=700, burn=250, seed=7 + rep)
    rec = dict(job='H2H', P=P, N=Nn, kind=kind, rep=rep, method=model,
               secs=round(time.time() - t0, 1), **score(Om, R, mask, model))
    return rec


# ------------------------------------------------------------------ fault injection

def fault_data():
    from sklearn.datasets import load_breast_cancer
    X = load_breast_cancer().data
    c = np.array([X[X[:, j] > 0, j].min() / 2 for j in range(X.shape[1])])
    L = np.log(X + c * (X <= 0).any(axis=0))
    return (L - L.mean(0)) / L.std(0), L.std(0)


def inject(Z, sds, pi, seed):
    """Decimal-point faults: a value recorded 10 times too large or too small."""
    rng = np.random.default_rng(seed)
    mask = rng.random(Z.shape) < pi
    shift = np.log(10.0) * rng.choice([-1.0, 1.0], size=Z.shape) / sds[None, :]
    return np.where(mask, Z + shift, Z), mask


def fault_cases():
    cases = [('clean', 0.0, 0)]
    for pi in (0.02, 0.05):
        for s in range(5):
            cases.append(('fault', pi, s))
    return cases


def run_fault(case, pi, seed, model):
    Z, sds = fault_data()
    X, mask = (Z, np.zeros(Z.shape, bool)) if case == 'clean' else inject(Z, sds, pi, 500 + seed)
    t0 = time.time()
    R = sample(X, model=model, n_iter=700, burn=250, seed=11 + seed)
    Oh = R['Omega']
    d = np.sqrt(np.diag(Oh))
    rec = dict(job='FAULT', case=case, pi=pi, seed=seed, method=model,
               secs=round(time.time() - t0, 1), nu=float(R['nu']),
               pcor=(-Oh / np.outer(d, d)).tolist())
    if model != 'gauss':
        w = np.broadcast_to(R['tau_mean'], mask.shape)
        if mask.any():
            rec['detect_auc'] = float(roc_auc_score(mask.ravel(), -w.ravel()))
        rec['cells_disc'] = float(np.mean(w < 0.5))
        if case == 'fault' and seed == 0 and pi == 0.05:
            np.save(f'{DIAG_DIR}/fault_tau_{model}.npy', np.asarray(R['tau_mean']))
            np.save(f'{DIAG_DIR}/fault_mask.npy', mask)
    return rec


# ------------------------------------------------------------------ diagnostics

def run_diag(name):
    os.makedirs(DIAG_DIR, exist_ok=True)
    if name.startswith('chain'):
        c = int(name[5:])
        Om, X, mask = grid_data(40, 'cell', 0)
        iu = np.triu_indices(40, 1)
        edges = [(int(iu[0][i]), int(iu[1][i])) for i in np.argsort(-np.abs(Om[iu]))[:3]]
        zero = [(int(iu[0][i]), int(iu[1][i])) for i in np.where(Om[iu] == 0)[0][:1]]
        R = sample(X, model='dirichlet', n_iter=2000, burn=500, seed=100 + c, trace=True,
                   trace_idx=edges + zero, store_omega=(c == 0))
        out = dict(nu=R['nu_trace'], **{k: v for k, v in R['trace'].items()},
                   idx=np.array(edges + zero), truth=np.array([Om[j, k] for j, k in edges + zero]))
        if c == 0:
            out['draws'] = R['omega_draws'][::2]
            out['Om'] = Om
        np.savez_compressed(f'{DIAG_DIR}/{name}.npz', **out)
    elif name == 'gauss_draws':
        Om, X, mask = grid_data(40, 'cell', 0)
        R = sample(X, model='gauss', n_iter=2000, burn=500, seed=100, store_omega=True)
        np.savez_compressed(f'{DIAG_DIR}/{name}.npz', draws=R['omega_draws'][::2], Om=Om)
    elif name.startswith('nu_'):
        _, model, p = name.split('_')
        p = int(p)
        Om, X, mask = grid_data(p, 'clean', 0)
        ni, bn = iters(p)
        R = sample(X, model=model, n_iter=ni, burn=bn, seed=7)
        R2 = sample(X, model=model, n_iter=ni, burn=bn, seed=7, nu_grid=NU_WIDE,
                    nu_logprior=ga_2_01)
        np.savez_compressed(f'{DIAG_DIR}/{name}.npz', nu=R['nu_trace'], nu_prior=R2['nu_trace'])
    return dict(job='DIAG', name=name)


# ------------------------------------------------------------------ job list and runner

def key(j):
    return json.dumps(j, sort_keys=True)


def build(which):
    jobs = []
    models = ['gauss', 'classical', 'alternative', 'dirichlet']
    if 'FIXNU' in which:
        for p in (120, 80, 40, 20):
            for kind in ('cell', 'row', 'clean'):
                for rep in (range(10) if p <= 40 else range(5)):
                    for m in models:
                        jobs.append(dict(fn='grid', job='FIXNU', p=p, kind=kind, rep=rep, model=m))
    if 'NUSAMP' in which:
        for p in (120, 80):
            for kind in ('cell', 'row', 'clean'):
                for rep in range(5):
                    for m in models[1:]:
                        jobs.append(dict(fn='grid', job='NUSAMP', p=p, kind=kind, rep=rep, model=m))
    if 'NUPRIOR' in which:
        for p in (120, 80):
            for kind in ('cell', 'row', 'clean'):
                for rep in range(5):
                    for m in models[1:]:
                        jobs.append(dict(fn='grid', job='NUPRIOR', p=p, kind=kind, rep=rep, model=m))
    for jb in ('ALPHA', 'ALPHAFIX'):
        if jb in which:
            for p in (120, 80, 40):
                for kind in ('cell', 'clean'):
                    for rep in range(5):
                        jobs.append(dict(fn='grid', job=jb, p=p, kind=kind, rep=rep, model='dirichlet'))
    if 'H2H' in which:
        for (P, Nn) in H2H_DESIGNS[::-1]:
            for kind in ('clean', 'cell', 'row'):
                for rep in H2H_REPS:
                    for m in models:
                        jobs.append(dict(fn='h2h', P=P, Nn=Nn, kind=kind, rep=rep, model=m))
    if 'FAULT' in which:
        for (case, pi, s) in fault_cases():
            for m in models:
                jobs.append(dict(fn='fault', case=case, pi=pi, seed=s, model=m))
    if 'DIAG' in which:
        for name in ['chain0', 'chain1', 'chain2', 'chain3', 'gauss_draws',
                     'nu_classical_40', 'nu_classical_80', 'nu_classical_120',
                     'nu_dirichlet_40', 'nu_dirichlet_80', 'nu_dirichlet_120']:
            jobs.append(dict(fn='diag', name=name))
    return jobs


def execute(j):
    j = dict(j)
    fn = j.pop('fn')
    try:
        rec = {'grid': run_grid, 'h2h': run_h2h, 'fault': run_fault, 'diag': run_diag}[fn](**j)
    except Exception as e:                     # keep the pool alive, record the failure
        rec = dict(error=repr(e))
    rec['_key'] = key(dict(fn=fn, **j))
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='DIAG,FIXNU,NUPRIOR,ALPHA,ALPHAFIX,H2H,FAULT')
    ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(DIAG_DIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if 'error' not in r:
                done.add(r['_key'])
    jobs = [j for j in build(a.jobs.split(',')) if key(j) not in done]
    print(f'{len(done)} done, {len(jobs)} to run', flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(execute, jobs)):
            with open(OUT, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f'[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] {rec["_key"]} '
                  f'{rec.get("relfro", rec.get("error", ""))}', flush=True)
    print('ALL DONE', flush=True)
