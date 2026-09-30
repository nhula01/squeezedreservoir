"""Cross-task demonstration: for each task, (a) coarse scan over the squeezing controls, (b) gradient descent from
random initial settings, (c) scan over the fabricated parameters (g, kappa, eps) at zero squeezing, and (d) effective
device parameters at the unsqueezed and squeezing-optimized settings. Also effective-parameter lines vs each control."""
from common import *
from sqz import make_lorenz, make_nce, effective_parameters

TASKS = {'mg': MG, 'narma': NARMA, 'lorenz': make_lorenz(horizon=2), 'nce': make_nce()}
R3, RE3, D3 = np.linspace(0, 0.5, 6), np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 13)[:-1]
G_GRID, K_GRID, E_GRID = np.linspace(0.25, 0.9, 6), np.linspace(0.1, 0.6, 6), np.linspace(0.1, 0.3, 5)

def coarse_scan(task):
    L3 = np.zeros((6, 6, 12)); N3 = L3.copy(); M3 = L3.copy()
    for i, r in enumerate(R3):
        for j, re in enumerate(RE3):
            for k, d in enumerate(D3):
                L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(DEV, (r, re, d), task)
        log(task.name, 'scan', i, L3[i].min())
    return L3, N3, M3

def refab_scan(task):
    L = np.zeros((6, 6, 5)); NR = L.copy(); NM = L.copy()
    for i, g in enumerate(G_GRID):
        for j, k in enumerate(K_GRID):
            for l, e in enumerate(E_GRID):
                L[i, j, l], NR[i, j, l], NM[i, j, l] = evaluate(replace(DEV, g=g, kappa=k, eps=e), (0, 0, 0), task)
        log(task.name, 'refab', i, L[i].min())
    return L, NR, NM

if __name__ == '__main__':
    fp = os.path.join(DATA, 'tasks.npz')
    out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    rng = np.random.default_rng(21)
    for name, task in TASKS.items():
        if f'{name}_L3' not in out:
            if name == 'mg':
                sc = np.load(os.path.join(DATA, 'fig2_scans.npz')); L3, N3, M3 = sc['L3'], sc['N3'], sc['M3']
            elif name == 'narma':
                sc = np.load(os.path.join(DATA, 'supp_narma_scan.npz')); L3, N3, M3 = sc['L3'], sc['N3'], sc['M3']
            else:
                L3, N3, M3 = coarse_scan(task)
            out[f'{name}_L3'], out[f'{name}_N3'], out[f'{name}_M3'] = L3, N3, M3
            np.savez(fp, **out)
        if f'{name}_gd' not in out:
            if name == 'mg':
                f3 = np.load(os.path.join(DATA, 'fig3_mg.npz'), allow_pickle=True); gd = f3['final_gd'][:, :6]; curves = f3['curve_gd']; inits = f3['inits']
            elif name == 'narma':
                f3 = np.load(os.path.join(DATA, 'fig3_narma.npz'), allow_pickle=True); gd = f3['final_gd'][:, :6]; curves = f3['curve_gd']; inits = f3['inits']
            else:
                inits = random_init(rng, 4); gd = []; curves = np.zeros((4, 100))
                for k, th0 in enumerate(inits):
                    cnt = Counter(DEV, task, budget=100); gradient_descent(cnt, th0, 100)
                    tr = np.array(cnt.trace); ib = tr[:, 4].argmin(); gd.append(tr[ib, 1:7]); curves[k] = cnt.best_so_far(100)
                    log(name, 'gd', k, 'L0=%.4e -> %.4e' % (tr[0, 4], tr[ib, 4]))
                gd = np.array(gd)
            out[f'{name}_gd'], out[f'{name}_gdcurves'], out[f'{name}_inits'] = gd, curves, inits
            np.savez(fp, **out)
        if f'{name}_refab' not in out:
            L, NR, NM = refab_scan(task)
            out[f'{name}_refab'], out[f'{name}_refabNR'], out[f'{name}_refabNM'] = L, NR, NM
            np.savez(fp, **out)
        if f'{name}_eff0' not in out:
            L3 = out[f'{name}_L3']; gd = out[f'{name}_gd']
            i, j, k = np.unravel_index(L3.argmin(), L3.shape)
            cands = [np.array([R3[i], RE3[j], D3[k]])] + [g[:3] for g in gd]
            Ls = [L3.min()] + [g[3] for g in gd]
            best = cands[int(np.argmin(Ls))]
            e0 = effective_parameters(DEV, (0, 0, 0)); e1 = effective_parameters(DEV, best)
            out[f'{name}_best'] = best; out[f'{name}_bestL'] = min(Ls)
            out[f'{name}_eff0'] = np.array([[k_, v] for k_, v in e0.items()], dtype=object); out[f'{name}_eff1'] = np.array([[k_, v] for k_, v in e1.items()], dtype=object)
            log(name, 'best', best, min(Ls), 'eff', e1)
            np.savez(fp, **out)
    if 'effline_dth' not in out:
        keys = list(effective_parameters(DEV, (0, 0, 0)).keys())
        for nm, pts in (('dth', [(0.0, 0.2, d) for d in np.linspace(-np.pi, np.pi, 25)]),
                        ('re', [(0.0, re, -0.79) for re in np.linspace(0, 0.5, 21)]),
                        ('r', [(r, 0.0, 0.0) for r in np.linspace(0, 0.5, 21)]),
                        ('re_pi', [(0.0, re, np.pi) for re in np.linspace(0, 0.5, 21)])):
            arr = np.array([[effective_parameters(DEV, th)[k] for k in keys] for th in pts])
            out[f'effline_{nm}'] = arr; out[f'effpts_{nm}'] = np.array(pts)
            log('effline', nm)
        out['effkeys'] = np.array(keys)
        np.savez(fp, **out)
    log('done tasks')
