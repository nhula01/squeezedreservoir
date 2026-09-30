"""Revision 2: Lorenz-63 with a relaxed photon budget. The wide-box ordinary optimum of rev_bounds2.py sits at the photon
budget max_t <a^dag a> = 2.5 used for truncation reliability at N_c = 10. Here the budget is raised to 4 and the Fock cutoff to
N_c = 16 (checked at 20): (1) the ordinary device is re-optimized locally from the wide-box optimum (bounded Nelder-Mead,
60 evaluations); (2) squeezing is searched on the relaxed optimum by convergent-gradient descent (60 evaluations) from zero
squeezing and from the squeezing optimum of rev_bounds2_final.py; measurement-aware objective of rev_bounds2.py.
Output: data/rev_bounds2_lorenz_budget.npz"""
import rev_bounds2 as r2
import rev_bounds as rb
import rev_base2
from rev_common import *
from scipy.optimize import minimize
NC, NMAX2 = 16, 4.0
rb.NMAX = NMAX2; rev_base2.NMAX = NMAX2
rb.device.__defaults__ = (0.0, NC)

if __name__ == '__main__':
    name = 'lorenz'; task = TASKS[name]; fp = os.path.join(DATA, f'rev_bounds2_{name}_budget.npz')
    d = np.load(os.path.join(DATA, f'rev_bounds2_{name}.npz'), allow_pickle=True); q = np.load(os.path.join(DATA, f'rev_bounds2_{name}_sqf.npz'), allow_pickle=True)
    bord = rb.best_row(d['ord_trace']); out = {}
    L0 = r2.evaluate_meas(rb.device(bord[:6]), (0, 0, 0), task); log('ordinary optimum at N_c %d: L %.4e nmax %.2f' % (NC, L0[0], L0[2]))
    obj = rb.Obj(task, 60, False); u0 = rb.to_u(bord[:6])
    simplex = np.vstack([u0] + [np.clip(u0 + 0.04 * np.eye(6)[k] * (1 if u0[k] < 0.5 else -1), 0, 1) for k in range(6)])
    try: minimize(obj, u0, method='Nelder-Mead', bounds=[(0, 1)] * 6, options=dict(maxfev=10 ** 6, initial_simplex=simplex))
    except StopIteration: pass
    T = np.array(obj.trace); ok = T[:, 12] <= NMAX2; b = T[np.where(ok)[0][T[ok, 10].argmin()]]; out['ord_trace'] = T; out['ord_best'] = b
    log('relaxed ordinary', dict(zip(rb.PN, b[:6].round(3))), 'L %.4e nmax %.2f' % (b[10], b[12]))
    res = []
    for pd, th0 in ((0.0, np.zeros(3)), (float(q['best'][9]), q['best'][6:9])):
        f = rev_base2.Bud(rb.device(b[:6], pd), task, 60)
        try: gradient_descent_conv(f, th0, 60)
        except StopIteration: pass
        Tq = np.array(f.trace); okq = Tq[:, 5] <= NMAX2; j = np.where(okq)[0][Tq[okq, 3].argmin()]; res.append([pd, *Tq[j]])
        log('squeezing from', np.round(th0, 3), 'phi_d %.2f:' % pd, Tq[j, :3].round(3), 'L %.4e (%.1f%%) nmax %.2f' % (Tq[j, 3], 100 * (1 - Tq[j, 3] / b[10]), Tq[j, 5]))
    out['sq'] = np.array(res); best = min(res, key=lambda r: r[4])
    c0 = r2.evaluate_meas(rb.device(b[:6], 0.0, 20), (0, 0, 0), task); c1 = r2.evaluate_meas(rb.device(b[:6], best[0], 20), np.array(best[1:4]), task)
    out['chk20'] = np.array([c0, c1]); np.savez(fp, **out)
    log('N_c 20: ordinary %.4e, squeezed %.4e (%.1f%%)' % (c0[0], c1[0], 100 * (1 - c1[0] / c0[0])))
