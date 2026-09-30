"""Referee-driven numerics: (gap) Liouvillian gaps along the control lines; (box) extended control box at N_c=12 for the
two edge optima; (baselines) reservoir-free baselines and a 100-node ESN per task; (refab) refabrication grid extended to
eps=0.5; (device2) a second fabricated device (T_in=4) that is already good for Mackey-Glass; (seeds) headline gains
re-evaluated on new input realizations at the fixed optimum."""
from common import *
from sqz import make_lorenz, make_nce, mackey_glass, narma10, nce, lorenz63, Task, ridge_fit, effective_parameters
from run_tasks import TASKS, R3, RE3, D3, G_GRID, K_GRID

def liouvillian_gap(dev, theta):
    res = Reservoir(dev, theta)
    ev = np.linalg.eigvals(res.L0r)
    re_ = np.sort(-ev.real); re_ = re_[re_ > 1e-9]
    return re_[0]

def delays(u, k):
    X = np.column_stack([np.roll(u, i) for i in range(k)]); X[:k] = 0; return X

def esn_features(u, n=100, alpha=0.3, rho=0.9, sin=0.5, seed=0):
    rng = np.random.default_rng(seed); W = rng.standard_normal((n, n)); W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.uniform(-sin, sin, n); b = rng.uniform(-0.1, 0.1, n); x = np.zeros(n); X = np.zeros((len(u), n))
    uf = 2 * (u - u.min()) / (u.max() - u.min()) - 1
    for t in range(len(u)):
        x = (1 - alpha) * x + alpha * np.tanh(W @ x + Win * uf[t] + b); X[t] = x
    return X

def fit_eval(F, task, delta=1e-8):
    tr, te = task.split(); Xf = np.hstack([F, np.ones((len(F), 1))])
    w = ridge_fit(Xf[tr], task.y[tr], delta)
    L = 0.5 * np.mean((task.y[tr] - Xf[tr] @ w) ** 2); yte = task.y[te]
    return L, np.sqrt(np.mean((yte - Xf[te] @ w) ** 2)) / np.std(yte)

def baselines(task):
    u = task.u; out = {}
    out['ols4'] = fit_eval(delays(u, 4), task); out['ols10'] = fit_eval(delays(u, 10), task)
    D = delays(u, 10); Q = np.column_stack([D[:, i] * D[:, j] for i in range(10) for j in range(i, 10)])
    out['quad10'] = fit_eval(np.hstack([D, Q]), task)
    best = None
    for alpha in (0.1, 0.3, 1.0):
        for sin in (0.1, 0.5, 1.0):
            for delta in (1e-6, 1e-3):
                r_ = fit_eval(esn_features(u, alpha=alpha, sin=sin), task, delta=delta)
                if best is None or r_[0] < best[0]: best = (r_[0], r_[1], alpha, sin, delta)   # selected on training loss
    out['esn100'] = best[:2]; out['esn100_cfg'] = best[2:]
    return out

if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    fp = os.path.join(DATA, 'review.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    tk = np.load(os.path.join(DATA, 'tasks.npz'), allow_pickle=True)
    if part in ('gap', 'all') and 'gap_re' not in out:
        for nm, pts in (('re', [(0.0, re, -0.79) for re in np.linspace(0, 0.5, 21)]), ('dth', [(0.0, 0.2, d) for d in np.linspace(-np.pi, np.pi, 25)]), ('r', [(r, 0.0, 0.0) for r in np.linspace(0, 0.5, 21)])):
            out[f'gap_{nm}'] = np.array([liouvillian_gap(DEV, th) for th in pts]); out[f'gappts_{nm}'] = np.array(pts)
        log('gaps', out['gap_re'][[0, -1]], np.ptp(out['gap_dth']), out['gap_r'][[0, -1]]); np.savez(fp, **out)
    if part in ('baselines', 'all') and 'base_mg' not in out:
        for nm, task in TASKS.items():
            b = baselines(task); out[f'base_{nm}'] = np.array([[k, v[0], v[1]] for k, v in b.items() if k != 'esn100_cfg'], dtype=object); out[f'esncfg_{nm}'] = np.array(b['esn100_cfg']); log('baselines', nm, b)
        np.savez(fp, **out)
    if part in ('box', 'all') and 'box_nce' not in out:
        dev12 = replace(DEV, Nc=12); grid = np.linspace(0, 0.8, 9)
        for nm, task, dth, s in (('nce', TASKS['nce'], -np.pi / 2, 1.0), ('lorenzflip', TASKS['lorenz'], 2.618, -1.0)):
            dv = replace(dev12, s=s); L = np.zeros((9, 9)); NR = L.copy(); NM = L.copy()
            for i, r in enumerate(grid):
                for j, re in enumerate(grid):
                    L[i, j], NR[i, j], NM[i, j] = evaluate(dv, (r, re, dth), task)
                log('box', nm, i, L[i].min())
            i, j = np.unravel_index(L.argmin(), L.shape); best = (grid[i], grid[j], dth)
            conv = [evaluate(replace(dv, Nc=Nc), best, task) for Nc in (12, 16)]
            out[f'box_{nm}'] = L; out[f'boxNR_{nm}'] = NR; out[f'boxNM_{nm}'] = NM; out[f'boxbest_{nm}'] = np.array(best); out[f'boxconv_{nm}'] = np.array(conv)
            log('box best', nm, best, L.min(), conv); np.savez(fp, **out)
        out['box_grid'] = grid; np.savez(fp, **out)
    if part in ('refab', 'all') and 'refabx_mg' not in out:
        dev10 = replace(DEV, Nc=10); E2 = np.array([0.4, 0.5])
        for nm, task in TASKS.items():
            L = np.zeros((6, 6, 2)); NR = L.copy(); NM = L.copy()
            for i, g in enumerate(G_GRID):
                for j, k in enumerate(K_GRID):
                    for l, e in enumerate(E2):
                        L[i, j, l], NR[i, j, l], NM[i, j, l] = evaluate(replace(dev10, g=g, kappa=k, eps=e), (0, 0, 0), task)
            out[f'refabx_{nm}'] = L; out[f'refabxNR_{nm}'] = NR; out[f'refabxNM_{nm}'] = NM; log('refab ext', nm, L.min(), NM.max()); np.savez(fp, **out)
        out['refabx_eps'] = E2; np.savez(fp, **out)
    if part in ('device2', 'all') and 'dev2_L3' not in out:
        dev2 = replace(DEV, T_in=4.0); L3 = np.zeros((6, 6, 12)); N3 = L3.copy(); M3 = L3.copy()
        for i, r in enumerate(R3):
            for j, re in enumerate(RE3):
                for k, d in enumerate(D3):
                    L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(dev2, (r, re, d), MG)
            log('device2 scan', i, L3[i].min())
        out['dev2_L3'], out['dev2_N3'], out['dev2_M3'] = L3, N3, M3
        rng = np.random.default_rng(31); fin = []
        for th0 in random_init(rng, 3):
            cnt = Counter(dev2, MG, budget=60); gradient_descent(cnt, th0, 60); tr = np.array(cnt.trace); ib = tr[:, 4].argmin(); fin.append(tr[ib, 1:7]); log('device2 gd', tr[0, 4], tr[ib, 1:5])
        out['dev2_gd'] = np.array(fin); np.savez(fp, **out)
    if part in ('seeds', 'all') and 'seeds_mg' not in out:
        real = {'mg': [Task('mg', x, np.roll(x, -10), 100, 700, 10) for x in (mackey_glass(1000, x0=0.9, transient=1500), mackey_glass(1000, x0=1.5, transient=2500))],
                'narma': [Task('narma10', *narma10(1000, s), 10, 590, 0) for s in (1, 2)],
                'lorenz': [Task('lorenz', x, np.roll(x, -2), 100, 700, 2) for x in (lorenz63(1000, transient=2600), lorenz63(1000, transient=3300))],
                'nce': [Task('nce', *nce(1000, s), 10, 690, 0) for s in (1, 2)]}
        for nm, tasks in real.items():
            best = tk[f'{nm}_best']; rows = []
            for T in tasks:
                a = evaluate(DEV, (0, 0, 0), T); b = evaluate(DEV, best, T); rows.append([a[0], b[0], a[1], b[1]])
            out[f'seeds_{nm}'] = np.array(rows); log('seeds', nm, rows)
        lf = np.load(os.path.join(DATA, 'lorenz_flip.npz'), allow_pickle=True); rows = []
        for T in real['lorenz']:
            a = evaluate(DEV, (0, 0, 0), T); b = evaluate(replace(DEV, s=-1.0), lf['best'], T); rows.append([a[0], b[0], a[1], b[1]])
        out['seeds_lorenzflip'] = np.array(rows); np.savez(fp, **out)
    log('done review', part)
