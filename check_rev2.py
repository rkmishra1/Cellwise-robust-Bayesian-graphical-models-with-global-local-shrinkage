"""Self-checks for the rev2 sampler additions. Defaults are unchanged and the alpha update
recovers a known alpha from simulated allocations."""
import numpy as np
import dtghs
import dtghs_pre_rev2 as old

rng = np.random.default_rng(0)
Om, _ = dtghs.make_precision(12, seed=1)
X = rng.standard_normal((60, 12)) @ np.linalg.cholesky(np.linalg.inv(Om)).T
for m in ['classical', 'alternative', 'dirichlet']:
    a = dtghs.sample(X, model=m, n_iter=60, burn=20, seed=3)['Omega']
    b = old.sample(X, model=m, n_iter=60, burn=20, seed=3)['Omega']
    assert np.allclose(a, b), m

K, p, n, a_true = 10, 40, 300, 2.0
pis = rng.dirichlet(np.full(K, a_true / K), size=n)
Z = np.array([rng.choice(K, p=pi, size=p) for pi in pis])
draws = [dtghs.update_alpha(Z, K, p, rng) for _ in range(400)]
lo, hi = np.quantile(draws, [0.025, 0.975])
print(f"alpha posterior 95% interval [{lo:.2f}, {hi:.2f}] (truth {a_true})")
assert lo < a_true < hi
print("ok")
