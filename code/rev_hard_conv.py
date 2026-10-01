"""Revision 2: convergence test for a large squeezing gain on a re-fabricated device (rev_hard.py). The lesson of rev_bounds2 is
that squeezing gains on an ordinary device that is not fully optimized disappear when the ordinary search is continued. Here
the ordinary search is given 450 more evaluations: Nelder-Mead restarts from the five best mutually distinct points of the
first search (60 each) and from the mapped effective device of the squeezed optimum (150), measurement-aware objective. The
squeezing search is then repeated on the improved ordinary optimum (206 evaluations).
Usage: python3 rev_hard_conv.py mg20 ...  -> data/rev_hard_<name>_conv.npz"""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *
from rev_hard import LADDER
from scipy.optimize import minimize

def nm(obj, u0, step, cap):
    full = obj.budget; obj.budget = min(cap, full)
    simplex = np.vstack([u0] + [np.clip(u0 + step * np.eye(6)[k] * (1 if u0[k] < 0.5 else -1), 0, 1) for k in range(6)])
    try: minimize(obj, u0, method='Nelder-Mead', bounds=[(0, 1)] * 6, options=dict(maxfev=10 ** 6, initial_simplex=simplex, xatol=1e-5, fatol=1e-12))
    except StopIteration: pass
    finally: obj.budget = full

if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        base, mk = LADDER[name]; task = mk(); h = np.load(os.path.join(DATA, f'rev_hard_{name}.npz'), allow_pickle=True)
        T = h['ord_trace']; bord = rb.best_row(T); sq = h['sq_best']
        obj = rb.Obj(task, 450, False)
        pen = np.where(T[:, 12] <= NMAX, T[:, 10], np.inf); U = np.array([rb.to_u(p) for p in T[:, :6]]); starts = []
        for i in np.argsort(pen):
            if all(np.linalg.norm(U[i] - U[j]) > 0.1 for j in starts): starts.append(i)
            if len(starts) == 5: break
        for m, i in enumerate(starts): nm(obj, U[i], 0.05, obj.n + 60)
        e = effective_parameters(rb.device(sq[:6], sq[9]), sq[6:9])                  # mapped effective device of the squeezed optimum
        pm = np.clip([e['g_co'], sq[1], sq[2], e['Omega'], sq[4], e['eps_eff']], rb.LOW, rb.HIW)
        nm(obj, rb.to_u(pm), 0.08, 450)
        T2 = np.array(obj.trace); b2 = rb.best_row(np.vstack([T, T2]))
        log(name, 'ordinary: first search %.4e, after +%d evaluations %.4e (%.1f%% lower)' % (bord[10], len(T2), b2[10], 100 * (1 - b2[10] / bord[10])))
        pd, th, L, S, n = rb.squeeze_on(rb.device(b2[:6]), task, b2[10])
        c0 = rb.check(rb.device(b2[:6]), (0, 0, 0), task, b2[12]); c1 = rb.check(rb.device(b2[:6], pd), th, task, L[2])
        np.savez(os.path.join(DATA, f'rev_hard_{name}_conv.npz'), trace=T2, ord_best=b2, sq_best=np.array([*b2[:6], *th, pd, *L]), chk0=c0, chk=c1)
        log(name, 'squeezing on the improved ordinary optimum', th.round(3), 'phi_d %.2f: checked %.4e -> %.4e (%.1f%%); squeezed first device %.4e' % (
            pd, c0[-1][0], c1[-1][0], 100 * (1 - c1[-1][0] / c0[-1][0]), h['chk'][-1][0]))
