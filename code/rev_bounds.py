"""Revision 2: is the squeezing gain on the re-fabricated device a boundary artefact?

The re-fabrication of rev_refab.py searched (g, kappa, gamma, omega_a, omega_q, eps) in a box whose faces several optima
touch (NARMA10 g = 1.20, Lorenz eps = 1.596, channel equalization kappa = 0.99, Mackey-Glass omega_a, omega_q = 2.0), and
squeezing rescales g_co = g cosh r beyond the face. This script
  (0) sanity check: the old re-fabricated optimum with g raised to the g_co of its squeezed optimum and the drive
      (eps up to 5, common detuning) re-tuned, no squeezing;
  (1) ORDINARY: re-fabrication in a much wider box (g 0.02-3, kappa 0.01-3, gamma 0.005-2, omega_a, omega_q 0-3,
      eps 0.01-5), Latin hypercube + multi-start Nelder-Mead, 706 evaluations; best at 500 evaluations is the start of
      the sequential search so that ordinary and sequential use the same total budget;
  (2) SEQUENTIAL: the Sec. 2.4 squeezing search (2 x 72 scan points + 60-evaluation convergent descent) from the
      wide-box optimum found with 500 evaluations (500 + 206 = 706 evaluations in total);
  (3) MATCHED: squeezing on the re-fabricated device with the three linear effective parameters held at their
      re-fabricated values, omega_a -> omega_a cosh 2r, g -> g / cosh r, eps -> eps / sqrt(cosh 2r - s sinh 2r cos 2 phi_d),
      so only the counter-rotating coupling g_cr and phase-sensitive damping can act;
  (4) MAPPED: an unsqueezed device given the effective parameters of the sequential optimum, then locally re-optimized
      over all six ordinary parameters in the wide box (120 evaluations);
  (5) JOINT: (6 ordinary + r, r_e, dtheta, phi_d) searched together from scratch, same protocol, 706 evaluations.
All optima are checked at N_c = 14 (N_c = 18 when the photon number is within 20% of the budget).
Usage: python3 rev_bounds.py narma lorenz ...   -> data/rev_bounds_<task>.npz"""
from rev_common import *
from rev_refab import PN
from rev_base2 import Bud, R, D8
from scipy.stats import qmc
from scipy.optimize import minimize
NC = 10
LOW = np.array([0.02, 0.01, 0.005, 0.0, 0.0, 0.01]); HIW = np.array([3.0, 3.0, 2.0, 3.0, 3.0, 5.0])
LOGW = np.array([True, True, True, False, False, True])
SQLO = np.array([0.0, 0.0, -np.pi, 0.0]); SQHI = np.array([0.5, 0.5, np.pi, np.pi])
B_MAIN, B_TOT, N_LHS, N_START = 500, 706, 200, 3

def to_p(u):
    u = np.clip(np.asarray(u, float), 0, 1); p = LOW + (HIW - LOW) * u
    p[LOGW] = LOW[LOGW] * (HIW[LOGW] / LOW[LOGW]) ** u[LOGW]; return p

def to_u(p):
    p = np.asarray(p, float); u = (p - LOW) / (HIW - LOW)
    u[LOGW] = np.log(p[LOGW] / LOW[LOGW]) / np.log(HIW[LOGW] / LOW[LOGW]); return u

def sq_p(v): return SQLO + (SQHI - SQLO) * np.clip(v, 0, 1)
def sq_u(q): return (np.asarray(q, float) - SQLO) / (SQHI - SQLO)

def device(p, phi_d=0.0, Nc=NC): return replace(DEV, Nc=Nc, phi_d=phi_d, **dict(zip(PN, p)))

class Obj:
    """Counted objective on the unit cube; d = 6 (ordinary) or 10 (joint). Trace rows: 6 params, r, re, dth, phi_d, L, NRMSE, nmax."""
    def __init__(self, task, budget, joint):
        self.task, self.budget, self.joint, self.n, self.trace = task, budget, joint, 0, []
    def __call__(self, u):
        if self.n >= self.budget: raise StopIteration
        p = to_p(u[:6]); q = sq_p(u[6:]) if self.joint else np.zeros(4)
        L, nr, nm = evaluate(device(p, q[3]), q[:3], self.task); self.n += 1; self.trace.append([*p, *q, L, nr, nm])
        return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))

def best_row(tr, upto=None):
    tr = np.asarray(tr)[:upto]; ok = tr[:, 12] <= NMAX; i = np.where(ok)[0][tr[ok, 10].argmin()]; return tr[i]

def global_search(obj, d, seeds, seed, main, total):
    """Latin hypercube (N_LHS) + seeds, Nelder-Mead from N_START distinct best points up to `main` evaluations,
    then Nelder-Mead continuation from the incumbent up to `total`."""
    U = np.vstack([np.atleast_2d(seeds), qmc.LatinHypercube(d=d, seed=seed).random(N_LHS)])
    vals = []
    for u in U: vals.append(obj(u))
    vals = np.array(vals); starts = []
    for i in np.argsort(vals):
        if all(np.linalg.norm(U[i] - U[j]) > 0.2 for j in starts): starts.append(i)
        if len(starts) == N_START: break
    per = (main - obj.n) // N_START
    def nm(u0, step, cap):
        simplex = np.vstack([u0] + [np.clip(u0 + step * np.eye(d)[k] * (1 if u0[k] < 0.5 else -1), 0, 1) for k in range(d)])
        full = obj.budget; obj.budget = min(cap, full)
        try:
            minimize(obj, u0, method='Nelder-Mead', bounds=[(0, 1)] * d,
                     options=dict(maxfev=10 ** 6, initial_simplex=simplex, xatol=1e-5, fatol=1e-12))
        except StopIteration:
            pass
        finally:
            obj.budget = full
    for m, i in enumerate(starts):
        nm(U[i], 0.08, main if m == len(starts) - 1 else obj.n + per)
    while obj.n < total:                              # continuation from the incumbent, restarted if NM converges early
        tr = np.array(obj.trace); pen = np.where(tr[:, 12] <= NMAX, tr[:, 10], np.inf); ib = pen.argmin()
        u0 = np.concatenate([to_u(tr[ib, :6]), sq_u(tr[ib, 6:10])[:d - 6]]) if d > 6 else to_u(tr[ib, :6])
        n0 = obj.n; nm(np.clip(u0, 0, 1), 0.03, total)
        if obj.n == n0: break

def squeeze_on(dev, task, L0, gd=60):
    """Sec. 2.4 protocol: 3x3x8 scans for phi_d = 0, pi/2, convergent descent from the best admissible scan point."""
    scans = {}
    for pd in (0.0, np.pi / 2):
        S = np.zeros((3, 3, 8, 3))
        for i, r in enumerate(R):
            for j, re in enumerate(R):
                for k, dd in enumerate(D8): S[i, j, k] = evaluate(replace(dev, phi_d=pd), (r, re, dd), task)
        S[..., 0] = np.where(S[..., 2] <= NMAX, S[..., 0], np.inf); scans[pd] = S
    pd = min(scans, key=lambda p: scans[p][..., 0].min()); S = scans[pd]
    i, j, k = np.unravel_index(S[..., 0].argmin(), S.shape[:3]); th0 = np.array([R[i], R[j], D8[k]])
    f = Bud(replace(dev, phi_d=pd), task, gd)
    try: gradient_descent_conv(f, th0, gd)
    except StopIteration: pass
    T = np.array(f.trace); T = T[T[:, 5] <= NMAX]
    th = T[T[:, 3].argmin(), :3] if len(T) and T[:, 3].min() < S[..., 0].min() else th0
    L = evaluate(replace(dev, phi_d=pd), th, task)
    return pd, th, np.array(L), np.stack([scans[0.0], scans[np.pi / 2]]), 2 * 72 + f.n

class Matched(Bud):
    """Squeezed device whose linear effective parameters (Omega, g_co, eps_eff) are pinned to those of dev0 at r = 0."""
    def __init__(self, dev0, task, budget=None):
        super().__init__(dev0, task, budget); self.dev0 = dev0
    def dev_at(self, th):
        r = th[0]; d0 = self.dev0
        fac = np.sqrt(np.cosh(2 * r) - d0.s * np.sinh(2 * r) * np.cos(2 * d0.phi_d))
        return replace(d0, omega=d0.omega * np.cosh(2 * r), g=d0.g / np.cosh(r), eps=d0.eps / fac)
    def __call__(self, th):
        if self.budget is not None and self.n >= self.budget: raise StopIteration
        th = project(th); L, nr, nm = evaluate(self.dev_at(th), th, self.task); self.n += 1; self.trace.append([*th, L, nr, nm])
        return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))

def matched_search(dev, task, gd=60):
    best = None; scans = []
    for pd in (0.0, np.pi / 2):
        f = Matched(replace(dev, phi_d=pd), task); S = np.zeros((3, 3, 8, 3))
        for i, r in enumerate(R):
            for j, re in enumerate(R):
                for k, dd in enumerate(D8):
                    f((r, re, dd)); S[i, j, k] = f.trace[-1][3:]
        S[..., 0] = np.where(S[..., 2] <= NMAX, S[..., 0], np.inf); scans.append(S)
        if best is None or S[..., 0].min() < best[0]:
            i, j, k = np.unravel_index(S[..., 0].argmin(), S.shape[:3]); best = (S[..., 0].min(), pd, np.array([R[i], R[j], D8[k]]))
    _, pd, th0 = best
    f = Matched(replace(dev, phi_d=pd), task, gd)
    try: gradient_descent_conv(f, th0, gd)
    except StopIteration: pass
    T = np.array(f.trace); T = T[T[:, 5] <= NMAX]
    th = T[T[:, 3].argmin(), :3] if len(T) and T[:, 3].min() < best[0] else th0
    dv = f.dev_at(th)
    return pd, th, np.array(evaluate(dv, th, task)), dv, np.stack(scans), 144 + f.n

def mapped_reopt(dsq, th, task, budget=120):
    """Unsqueezed device with the effective parameters of the squeezed optimum (Omega, g_co, eps_eff; bare gamma chosen to
    reproduce the mean effective emitter relaxation rate), then Nelder-Mead over all six ordinary parameters."""
    e = effective_parameters(dsq, th)
    lin = replace(dsq, omega=e['Omega'], g=e['g_co'], eps=e['eps_eff'], phi_d=0.0)
    target = np.mean([e['gamma_x'], e['gamma_y'], e['gamma_z']])
    def gmean(gm):
        ee = effective_parameters(replace(lin, gamma=gm), (0, 0, 0)); return np.mean([ee['gamma_x'], ee['gamma_y'], ee['gamma_z']])
    lo, hi = LOW[2], HIW[2]
    for _ in range(14):
        mid = np.sqrt(lo * hi); lo, hi = (mid, hi) if gmean(mid) < target else (lo, mid)
    p0 = np.clip([lin.g, lin.kappa, np.sqrt(lo * hi), lin.omega, lin.omega_q, lin.eps], LOW, HIW)
    obj = Obj(task, budget, False); obj(to_u(p0)); Lmap = obj.trace[0][10:13]
    u0 = to_u(p0); simplex = np.vstack([u0] + [np.clip(u0 + 0.05 * np.eye(6)[k] * (1 if u0[k] < 0.5 else -1), 0, 1) for k in range(6)])
    try:
        minimize(obj, u0, method='Nelder-Mead', bounds=[(0, 1)] * 6, options=dict(maxfev=10 ** 6, initial_simplex=simplex, xatol=1e-5, fatol=1e-12))
    except StopIteration:
        pass
    return p0, np.array(Lmap), np.array(obj.trace), e

def check(dev, th, task, nm):
    out = [evaluate(replace(dev, Nc=14), th, task)]
    if nm > 0.8 * NMAX: out.append(evaluate(replace(dev, Nc=18), th, task))
    return np.array(out)

if __name__ == '__main__':
    which = sys.argv[1:] or list(TASKS)
    rf = np.load(os.path.join(DATA, 'rev_refab.npz'), allow_pickle=True)
    rs = np.load(os.path.join(DATA, 'rev_refab_sq.npz'), allow_pickle=True)
    for k, name in enumerate(which):
        task = TASKS[name]; fp = os.path.join(DATA, f'rev_bounds_{name}.npz')
        out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
        save = lambda: np.savez(fp, **out)
        old = rf[f'{name}_best']; p_old = old[:6]; seed = 300 + list(TASKS).index(name)
        # (0) sanity check
        if 'sanity' not in out:
            r_sq = float(rs[f'{name}_best'][1]); g1 = p_old[0] * np.cosh(r_sq); tr = []
            def fs(x):
                e = float(np.clip(np.exp(x[0]), LOW[5], HIW[5])); dl = float(np.clip(x[1], -min(p_old[3], p_old[4]), 1.0))
                dv = device([g1, p_old[1], p_old[2], p_old[3] + dl, p_old[4] + dl, e]); L, nr, nm = evaluate(dv, (0, 0, 0), task)
                tr.append([g1, e, dl, L, nr, nm]); return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))
            x0 = np.array([np.log(p_old[5]), 0.0])
            minimize(fs, x0, method='Nelder-Mead', options=dict(maxfev=40, initial_simplex=[x0, x0 + [0.25, 0], x0 + [0, 0.1]]))
            tr = np.array(tr); ok = tr[:, 5] <= NMAX; out['sanity'] = tr; out['sanity_best'] = tr[ok][tr[ok, 3].argmin()]
            log(name, 'sanity g %.3f -> %.3f, drive re-tuned: L %.4e (old refab %.4e, old refab+sq %.4e)' % (
                p_old[0], g1, out['sanity_best'][3], old[6], rs[f'{name}_Lsq'][0][0])); save()
        # (1) ordinary, wide box
        if 'ord_trace' not in out:
            obj = Obj(task, B_TOT, False); global_search(obj, 6, [to_u(p_old)], seed, B_MAIN, B_TOT)
            out['ord_trace'] = np.array(obj.trace); save()
        T = out['ord_trace']; b500 = best_row(T, B_MAIN); bord = best_row(T)
        uo = to_u(bord[:6]); out['ord_best'] = bord; out['ord_best500'] = b500
        out['ord_interior'] = np.array([min(x, 1 - x) for x in uo])
        log(name, 'ORDINARY wide', dict(zip(PN, bord[:6].round(3))), 'L %.4e (500: %.4e; old %.4e) nmax %.2f min face dist %.3f' % (
            bord[10], b500[10], old[6], bord[12], out['ord_interior'].min()))
        # (2) sequential: squeezing on the 500-evaluation ordinary optimum
        if 'seq_best' not in out:
            dv = device(b500[:6]); pd, th, L, S, n = squeeze_on(dv, task, b500[10])
            out['seq_best'] = np.array([*b500[:6], *th, pd, *L]); out['seq_scans'] = S; out['seq_evals'] = np.array(B_MAIN + n)
            log(name, 'SEQUENTIAL', th.round(3), 'phi_d %.2f L %.4e -> %.4e (%.1f%%) nmax %.2f' % (pd, b500[10], L[0], 100 * (1 - L[0] / b500[10]), L[2])); save()
        # (3) matched effective parameters, on the final ordinary optimum
        if 'mat_best' not in out:
            dv = device(bord[:6]); pd, th, L, dvm, S, n = matched_search(dv, task)
            out['mat_best'] = np.array([dvm.g, dvm.kappa, dvm.gamma, dvm.omega, dvm.omega_q, dvm.eps, *th, pd, *L]); out['mat_scans'] = S
            log(name, 'MATCHED (Omega, g_co, eps_eff pinned)', th.round(3), 'phi_d %.2f L %.4e -> %.4e (%.1f%%)' % (pd, bord[10], L[0], 100 * (1 - L[0] / bord[10]))); save()
        # (4) mapped device, locally re-optimized
        if 'map_trace' not in out:
            sb = out['seq_best']; dsq = device(sb[:6], sb[9])
            p0, Lmap, trm, e = mapped_reopt(dsq, sb[6:9], task)
            out['map_p0'] = p0; out['map_L0'] = Lmap; out['map_trace'] = trm; out['map_eff'] = np.array([[a, b] for a, b in e.items()], dtype=object)
            bm = best_row(trm); log(name, 'MAPPED device L %.4e, re-optimized %.4e (seq %.4e)' % (Lmap[0], bm[10], sb[10])); save()
        # (5) joint search
        if 'joint_trace' not in out:
            obj = Obj(task, B_TOT, True); global_search(obj, 10, [np.concatenate([to_u(p_old), np.zeros(4)])], seed + 50, B_MAIN, B_TOT)
            out['joint_trace'] = np.array(obj.trace); save()
        bj = best_row(out['joint_trace']); out['joint_best'] = bj
        log(name, 'JOINT', dict(zip(PN, bj[:6].round(3))), 'sq', bj[6:10].round(3), 'L %.4e nmax %.2f' % (bj[10], bj[12]))
        # convergence checks
        if 'checks' not in out:
            sb, mb = out['seq_best'], out['mat_best']
            out['chk_ord'] = check(device(bord[:6]), (0, 0, 0), task, bord[12])
            out['chk_seq'] = check(device(sb[:6], sb[9]), sb[6:9], task, sb[12])
            out['chk_mat'] = check(device(mb[:6], mb[9]), mb[6:9], task, mb[12])
            out['chk_joint'] = check(device(bj[:6], bj[9]), bj[6:9], task, bj[12])
            out['checks'] = np.array(1)
            log(name, 'Nc checks: ord', out['chk_ord'][:, 0], 'seq', out['chk_seq'][:, 0], 'mat', out['chk_mat'][:, 0], 'joint', out['chk_joint'][:, 0])
        save()
    log('done bounds', which)
