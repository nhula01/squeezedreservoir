"""Concern 2a/2b: training the squeezing controls from measured, noisy losses. revised base device (drive amplitude and frequency
optimized) of the task with the largest squeezing gain,
start with the squeezers off (r = r_e = 0). Finite-difference gradient descent (projected, backtracking, the optimizer of
the main text) and SPSA at matched evaluation budgets, for N_rep in NREP passes per setting, and once more with all
non-idealities on (injection efficiency 0.7, fast phase jitter 0.1 rad, slow drift). The clean loss of every evaluated
setting is recorded next to the noisy loss the optimizer sees."""
from rev_common import *
NC = 10; BUDGET = 48; SEEDS = 2
NREP = [1e5, 1e6, 1e7]
H_OF = {1e5: 0.07, 1e6: 0.05, 1e7: 0.035}            # h* ~ N^{-1/6} (Supplementary Note 12)
DRIFT = dict(r=0.01, re=0.01, eps=0.01, dth=0.02, phid=0.02)

def spsa(fun, theta0, budget, c, rng, a0=0.15, A=3):
    z = project(theta0) / SCALE; k = 0; path = []; a = None
    while True:
        ck = c / (k + 1) ** 0.101; d = rng.choice([-1.0, 1.0], 3)
        try:
            Lp = fun(project((z + ck * d) * SCALE)); Lm = fun(project((z - ck * d) * SCALE))
        except StopIteration:
            break
        g = (Lp - Lm) / (2 * ck) * d
        if a is None: a = a0 * (A + 1) ** 0.602 / max(np.abs(g).mean(), 1e-12)
        z = project((z - a / (k + 1 + A) ** 0.602 * g) * SCALE) / SCALE; k += 1; path.append(z * SCALE)
    return np.array(path)

def run(method, N, seed, nonideal=False):
    NAME = training_task(); task = TASKS[NAME]; dev, _ = squeezed_point2(NAME, NC); rng = np.random.default_rng(1000 * seed + int(np.log10(N)) + (50 if nonideal else 0))
    kw = dict(eta_inj=0.7, sig_phi=0.1, drift=DRIFT) if nonideal else {}
    f = ExpLoss(dev, task, Nrep=N, rng=rng, budget=BUDGET, project=project, **kw)
    h = H_OF[N]; th0 = np.zeros(3)
    if method == 'gd':
        path = gradient_descent(f, th0, BUDGET, h=np.array([h, h, h]))
        final = path[-1][0]
    else:
        p = spsa(f, th0, BUDGET, c=2 * h, rng=rng); final = p[-1] if len(p) else th0
    Lfinal = evaluate(dev, final, task)[0]                                                      # ideal device, clean
    Lfinal_ni = ExpLoss(dev, task, Nrep=None, eta_inj=kw.get('eta_inj', 1.0), sig_phi=kw.get('sig_phi', 0.0))(final)  # non-ideal bath, clean readout
    return np.array(f.trace), final, Lfinal, Lfinal_ni, f.passes

if __name__ == '__main__':
    fp = os.path.join(DATA, 'rev_train.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}; out.pop('task', None)
    jobs = [(m, N, s, False) for N in NREP for m in ('gd', 'spsa') for s in range(SEEDS)] + [(m, 1e6, s, True) for m in ('gd', 'spsa') for s in range(SEEDS)]
    for m, N, s, ni in jobs:
        key = f'{m}_{int(np.log10(N))}_{s}' + ('_ni' if ni else '')
        if key + '_trace' in out: continue
        tr, final, Lf, Lfni, passes = run(m, N, s, ni)
        out[key + '_trace'] = tr; out[key + '_final'] = final; out[key + '_Lfinal'] = Lf; out[key + '_Lfinal_ni'] = Lfni; out[key + '_passes'] = passes
        log(key, 'final', final.round(3), 'clean L %.4e (ni-bath %.4e)' % (Lf, Lfni), 'passes %.2e' % passes)
        np.savez(fp, task=training_task(), **out)
    log('done train')
