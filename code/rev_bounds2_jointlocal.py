"""Revision 2 helper: where the joint search ends below the ordinary optimum, is that squeezing or an ordinary device the
ordinary search missed? The device of the joint optimum is re-optimized locally at zero squeezing over the six ordinary
parameters (bounded Nelder-Mead, 150 evaluations, measurement-aware objective) and compared with the joint optimum.
Output: data/rev_bounds2_<task>_jointlocal.npz"""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *
from scipy.optimize import minimize
if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        d = np.load(os.path.join(DATA, f'rev_bounds2_{name}.npz'), allow_pickle=True); j = rb.best_row(d['joint_trace']); task = TASKS[name]
        obj = rb.Obj(task, 150, False); u0 = rb.to_u(j[:6])
        simplex = np.vstack([u0] + [np.clip(u0 + 0.05 * np.eye(6)[k] * (1 if u0[k] < 0.5 else -1), 0, 1) for k in range(6)])
        try: minimize(obj, u0, method='Nelder-Mead', bounds=[(0, 1)] * 6, options=dict(maxfev=10 ** 6, initial_simplex=simplex))
        except StopIteration: pass
        T = np.array(obj.trace); b = rb.best_row(T); c = rb.check(rb.device(b[:6]), (0, 0, 0), task, b[12])
        np.savez(os.path.join(DATA, f'rev_bounds2_{name}_jointlocal.npz'), trace=T, best=b, chk=c)
        log(name, 'joint optimum %.4e (squeezed); ordinary re-optimized from its device %.4e (checked %.4e); wide ordinary %.4e' % (j[10], b[10], c[-1][0], rb.best_row(d['ord_trace'])[10]))
