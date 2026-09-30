"""Concern 1: a re-fabrication baseline that optimizes every device parameter the squeezing controls mimic.
At zero squeezing, optimize (g, kappa, gamma, omega_a, omega_q, eps) under the photon budget of Section 2.5, with an
evaluation budget larger than the whole squeezing search of that section (445 evaluations): 160 Latin-hypercube points
(plus the drive-optimized base device) and Nelder-Mead from the three best, 100 evaluations each. omega_a and omega_q are
the cavity and emitter detunings from the drive carrier, so the search also contains every drive-frequency setting.
Then: (a) drive frequency and amplitude alone (post-fabrication, no squeezing): common shift of omega_a, omega_q;
(b) mapped effective device: an unsqueezed device given the effective parameters of each squeezed optimum."""
from rev_common import *
from scipy.stats import qmc
PN = ['g', 'kappa', 'gamma', 'omega', 'omega_q', 'eps']
LO = np.array([0.1, 0.05, 0.02, 0.3, 0.3, 0.05]); HI = np.array([1.2, 1.0, 0.6, 2.0, 2.0, 1.6])
LOG = np.array([True, True, True, False, False, True])
NC = 10

def to_p(u):
    u = np.clip(u, 0, 1)
    return np.where(LOG, LO * (HI / LO) ** u, LO + (HI - LO) * u)

def to_u(p):
    p = np.asarray(p, float)
    return np.where(LOG, np.log(p / LO) / np.log(HI / LO), (p - LO) / (HI - LO))

class Fab:
    def __init__(self, task, budget): self.task, self.budget, self.n, self.trace = task, budget, 0, []
    def __call__(self, u):
        if self.n >= self.budget: raise StopIteration
        p = to_p(u); dv = replace(DEV, Nc=NC, **dict(zip(PN, p)))
        L, nr, nm = evaluate(dv, (0, 0, 0), self.task); self.n += 1; self.trace.append([*p, L, nr, nm])
        return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))

def refab(name, task, seed):
    from scipy.optimize import minimize
    f = Fab(task, 300)
    U = qmc.LatinHypercube(d=6, seed=seed).random(119)
    b = base_device(name); U = np.vstack([to_u([b.g, b.kappa, b.gamma, b.omega, b.omega_q, b.eps]), U])
    vals = np.array([f(u) for u in U])
    order = np.argsort(vals); starts = []
    for i in order:                                   # three best, mutually distinct starts
        if all(np.linalg.norm(U[i] - U[j]) > 0.15 for j in starts): starts.append(i)
        if len(starts) == 3: break
    for i in starts:
        try:
            simplex = np.vstack([U[i]] + [np.clip(U[i] + 0.1 * np.eye(6)[k], 0, 1) for k in range(6)])
            minimize(f, U[i], method='Nelder-Mead', bounds=[(0, 1)] * 6,
                     options=dict(maxfev=60, initial_simplex=simplex, xatol=1e-4, fatol=1e-10))
        except StopIteration:
            pass
    tr = np.array(f.trace); ok = tr[:, 8] <= NMAX; ib = np.where(ok)[0][tr[ok, 6].argmin()]
    return tr, ib

def mapped(name, task):
    """Unsqueezed device with the effective linear parameters of the squeezed optimum (Omega, g cosh r, eps_eff),
    then additionally with the bare emitter decay chosen to reproduce the squeezed device's gamma_z^eff."""
    dsq, th = squeezed_point(name); dsq = replace(dsq, Nc=NC)
    e = effective_parameters(dsq, th)
    lin = replace(DEV, Nc=NC, omega=e['Omega'], g=e['g_co'], eps=e['eps_eff'])
    Llin = evaluate(lin, (0, 0, 0), task)
    target = np.mean([e['gamma_x'], e['gamma_y'], e['gamma_z']])
    def gmean(gm):
        ee = effective_parameters(replace(lin, gamma=gm), (0, 0, 0)); return np.mean([ee['gamma_x'], ee['gamma_y'], ee['gamma_z']])
    lo, hi = 1e-3, 2.0
    if gmean(lo) > target: gam = lo
    else:
        for _ in range(18):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if gmean(mid) < target else (lo, mid)
        gam = np.sqrt(lo * hi)
    full = replace(lin, gamma=gam)
    Lfull = evaluate(full, (0, 0, 0), task)
    Lsq = evaluate(dsq, th, task)
    return dict(eff=np.array([[k, v] for k, v in e.items()], dtype=object), Llin=np.array(Llin), Lfull=np.array(Lfull),
                gamma_map=gam, target=target, Lsq=np.array(Lsq))

if __name__ == '__main__':
    which = sys.argv[1:] or list(TASKS)
    fp = os.path.join(DATA, 'rev_refab.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    for k, name in enumerate(which):
        task = TASKS[name]
        if f'{name}_trace' not in out:
            tr, ib = refab(name, task, seed=100 + k)
            out[f'{name}_trace'] = tr; out[f'{name}_best'] = tr[ib]
            best = tr[ib]; dv = replace(DEV, Nc=14, **dict(zip(PN, best[:6])))
            out[f'{name}_bestconv'] = np.array(evaluate(dv, (0, 0, 0), task))
            log(name, 'REFAB best', dict(zip(PN, best[:6].round(3))), 'L %.4e (Nc14 %.4e) nmax %.2f' % (best[6], out[f'{name}_bestconv'][0], best[8]), 'evals', len(tr))
            np.savez(fp, **out)
        if f'{name}_drive' not in out:              # post-fabrication drive frequency + amplitude, no squeezing
            b = base_device(name, NC); rows = []
            for fe in (0.7, 1.0, 1.4):
                for dl in np.linspace(-0.6, 0.6, 9):
                    dv = replace(b, eps=b.eps * fe, omega=1 + dl, omega_q=1 + dl)
                    rows.append([b.eps * fe, dl, *evaluate(dv, (0, 0, 0), task)])
            rows = np.array(rows); out[f'{name}_drive'] = rows
            ok = rows[:, 4] <= NMAX; log(name, 'drive freq+amp best', rows[ok][rows[ok, 2].argmin()].round(4)); np.savez(fp, **out)
        if f'{name}_map' not in out:
            m = mapped(name, task)
            for kk, v in m.items(): out[f'{name}_map_{kk}'] = v
            out[f'{name}_map'] = np.array(1)
            log(name, 'mapped: sq %.4e  linear-map %.4e  +gamma(%.3f) %.4e' % (m['Lsq'][0], m['Llin'][0], m['gamma_map'], m['Lfull'][0])); np.savez(fp, **out)
    log('done refab', which)
