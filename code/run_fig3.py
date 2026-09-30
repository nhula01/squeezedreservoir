"""Fig. 3: gradient descent vs random search vs Nelder-Mead under a matched budget of reservoir evaluations."""
from common import *

def run(task, dev, n_init, budget, seed, tag):
    rng = np.random.default_rng(seed)
    inits = random_init(rng, n_init)
    curves = {m: np.zeros((n_init, budget)) for m in ('gd', 'rs', 'nm')}
    finals = {m: [] for m in curves}
    paths = []
    k0 = 0
    fp = os.path.join(DATA, f'fig3_{tag}.npz')
    if os.path.exists(fp):                       # resume
        old = np.load(fp, allow_pickle=True)
        k0 = len(old['final_gd'])
        for m in curves:
            curves[m][:k0] = old[f'curve_{m}'][:k0]; finals[m] = list(old[f'final_{m}'])
        paths = list(old['paths'])
        log(tag, 'resuming from init', k0)
    for k, th0 in enumerate(inits):
        if k < k0:
            continue
        # gradient descent
        cnt = Counter(dev, task, budget=budget)
        p = gradient_descent(cnt, th0, budget)
        curves['gd'][k] = cnt.best_so_far(budget)
        tr = np.array(cnt.trace); ib = tr[:, 4].argmin()
        finals['gd'].append(tr[ib, 1:7]); paths.append(np.array([[*t, L] for t, L, _ in p]))
        # random search (seeded per init; first sample is the same initial point for fairness)
        cnt = Counter(dev, task, budget=budget)
        cnt(th0); random_search(cnt, budget, np.random.default_rng(1000 + seed * 100 + k))
        curves['rs'][k] = cnt.best_so_far(budget)
        tr = np.array(cnt.trace); ib = tr[:, 4].argmin(); finals['rs'].append(tr[ib, 1:7])
        # Nelder-Mead
        cnt = Counter(dev, task, budget=budget)
        nelder_mead(cnt, th0, budget)
        curves['nm'][k] = cnt.best_so_far(budget)
        tr = np.array(cnt.trace); ib = tr[:, 4].argmin(); finals['nm'].append(tr[ib, 1:7])
        log(tag, 'init', k, 'L0=%.4e' % curves['gd'][k][0], 'gd=%.4e rs=%.4e nm=%.4e' % (curves['gd'][k][-1], curves['rs'][k][-1], curves['nm'][k][-1]))
        np.savez(os.path.join(DATA, f'fig3_{tag}.npz'), inits=inits, budget=budget,
                 **{f'curve_{m}': curves[m] for m in curves}, **{f'final_{m}': np.array(finals[m]) for m in finals},
                 paths=np.array(paths, dtype=object), allow_pickle=True)
    return curves, finals

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'mg'
    if which == 'mg':
        run(MG, DEV, n_init=12, budget=100, seed=3, tag='mg')
    else:
        run(NARMA, DEV, n_init=8, budget=80, seed=4, tag='narma')
    log('done fig3', which)
