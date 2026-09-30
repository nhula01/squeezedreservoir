"""Training v2: coarse pre-scan + finite-difference gradient descent (or SPSA) with the SNR-aware readout.
Same task, device, budgets and seeds as rev_train.py: 12 pre-scan evaluations (the origin plus 11 scrambled-Sobol settings
in the control box), then descent from the setting with the lowest measured loss for the remaining 36 evaluations."""
from rev_common import *
from rev_train import spsa, H_OF, DRIFT, NREP, SEEDS, BUDGET, NC
from scipy.stats import qmc
NPRE = 12

def run(method, N, seed, nonideal=False):
    NAME = training_task(); task = TASKS[NAME]; dev, _ = squeezed_point2(NAME, NC)
    rng = np.random.default_rng(2000 * seed + int(np.log10(N)) + (50 if nonideal else 0))
    kw = dict(eta_inj=0.7, sig_phi=0.1, drift=DRIFT) if nonideal else {}
    f = ExpLoss(dev, task, Nrep=N, rng=rng, budget=BUDGET, project=project, **kw); f.readout = 'snr'
    pts = np.vstack([np.zeros(3), qmc.scale(qmc.Sobol(3, seed=seed).random(16)[:NPRE - 1], [0, 0, -np.pi], [0.5, 0.5, np.pi])])
    Ls = [f(p) for p in pts]; th0 = pts[int(np.argmin(Ls))]; h = H_OF[N]
    if method == 'gd':
        path = gradient_descent(f, th0, BUDGET, h=np.array([h, h, h])); final = path[-1][0]
    else:
        p = spsa(f, th0, BUDGET, c=2 * h, rng=rng); final = p[-1] if len(p) else th0
    return np.array(f.trace), final, evaluate(dev, final, task)[0], f.passes

if __name__ == '__main__':
    fp = os.path.join(DATA, 'rev_train2.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}; out.pop('task', None)
    jobs = [(m, N, s, False) for N in NREP for m in ('gd', 'spsa') for s in range(SEEDS)] + [(m, 1e6, s, True) for m in ('gd', 'spsa') for s in range(SEEDS)]
    for m, N, s, ni in jobs:
        key = f'{m}_{int(np.log10(N))}_{s}' + ('_ni' if ni else '')
        if key + '_trace' in out: continue
        tr, final, Lf, passes = run(m, N, s, ni)
        out[key + '_trace'] = tr; out[key + '_final'] = final; out[key + '_Lfinal'] = Lf; out[key + '_passes'] = passes
        log(key, 'final', final.round(3), 'clean L %.4e' % Lf); np.savez(fp, task=training_task(), **out)
    log('done train2')
