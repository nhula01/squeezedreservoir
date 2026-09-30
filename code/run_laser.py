"""Real-world task: one-step prediction of the Santa Fe far-infrared laser intensity (experimental chaotic data)."""
from common import *
from sqz import Task, effective_parameters
from run_tasks import R3, RE3, D3, G_GRID, K_GRID, E_GRID
from run_review import baselines
import reservoirpy.datasets as ds
X = ds.santafe_laser().ravel().astype(float)
def make(start): x = X[start:start + 1000]; return Task('laser', x, np.roll(x, -1), 100, 700, 1)
LASER = make(0)
if __name__ == '__main__':
    fp = os.path.join(DATA, 'laser.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    if 'base' not in out:
        b = baselines(LASER); out['base'] = np.array([[k, v[0], v[1]] for k, v in b.items() if k != 'esn100_cfg'], dtype=object); log('baselines', b)
    if 'L3' not in out:
        L3 = np.zeros((6, 6, 12)); N3 = L3.copy(); M3 = L3.copy()
        for i, r in enumerate(R3):
            for j, re in enumerate(RE3):
                for k, d in enumerate(D3): L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(DEV, (r, re, d), LASER)
            log('scan', i, L3[i].min())
        out['L3'], out['N3'], out['M3'] = L3, N3, M3; np.savez(fp, **out)
    if 'gd' not in out:
        rng = np.random.default_rng(41); inits = random_init(rng, 4); gd = []; curves = np.zeros((4, 100))
        for k, th0 in enumerate(inits):
            cnt = Counter(DEV, LASER, budget=100); gradient_descent(cnt, th0, 100); tr = np.array(cnt.trace); ib = tr[:, 4].argmin(); gd.append(tr[ib, 1:7]); curves[k] = cnt.best_so_far(100)
            log('gd', k, tr[0, 4], '->', tr[ib, 1:5])
        out['gd'], out['gdcurves'], out['inits'] = np.array(gd), curves, inits; np.savez(fp, **out)
    if 'refab' not in out:
        L = np.zeros((6, 6, 7)); NR = L.copy(); NM = L.copy(); E = np.concatenate([E_GRID, [0.4, 0.5]])
        for i, g in enumerate(G_GRID):
            for j, k in enumerate(K_GRID):
                for l, e in enumerate(E):
                    L[i, j, l], NR[i, j, l], NM[i, j, l] = evaluate(replace(DEV, g=g, kappa=k, eps=e, Nc=8 if e <= 0.3 else 10), (0, 0, 0), LASER)
            log('refab', i, L[i].min())
        out['refab'], out['refabNR'], out['refabNM'], out['refab_eps'] = L, NR, NM, E; np.savez(fp, **out)
    if 'best' not in out:
        L3 = out['L3']; i, j, k = np.unravel_index(L3.argmin(), L3.shape); cands = [np.array([R3[i], RE3[j], D3[k]])] + [g[:3] for g in out['gd']]; Ls = [L3.min()] + [g[3] for g in out['gd']]
        best = cands[int(np.argmin(Ls))]; out['best'] = best; out['bestL'] = min(Ls)
        out['conv'] = np.array([evaluate(replace(DEV, Nc=Nc), best, LASER) for Nc in (8, 12, 16)])
        out['eff0'] = np.array([[k_, v] for k_, v in effective_parameters(DEV, (0, 0, 0)).items()], dtype=object); out['eff1'] = np.array([[k_, v] for k_, v in effective_parameters(DEV, best).items()], dtype=object)
        out['seeds'] = np.array([[evaluate(DEV, (0, 0, 0), T)[0], evaluate(DEV, best, T)[0], evaluate(DEV, (0, 0, 0), T)[1], evaluate(DEV, best, T)[1]] for T in (make(2500), make(5000))])
        log('best', best, min(Ls), 'conv', out['conv'], 'seeds', out['seeds']); np.savez(fp, **out)
    # amplifying branch coarse scan at reduced resolution
    if 'flipL3' not in out:
        dv = replace(DEV, s=-1.0); L3 = np.zeros((6, 6, 12)); N3 = L3.copy(); M3 = L3.copy()
        for i, r in enumerate(R3):
            for j, re in enumerate(RE3):
                for k, d in enumerate(D3): L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(dv, (r, re, d), LASER)
            log('flip scan', i, L3[i].min())
        out['flipL3'], out['flipN3'], out['flipM3'] = L3, N3, M3; np.savez(fp, **out)
    log('done laser')
