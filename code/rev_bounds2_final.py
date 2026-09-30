"""Revision 2: squeezing on the FINAL wide-box ordinary optimum of rev_bounds2.py (506 evaluations), with the same
measurement-aware objective and the Sec. 2.4 squeezing protocol (2 x 72 scan points + 60 convergent-descent evaluations).
rev_bounds2.py's 'sequential' stage squeezes the 300-evaluation optimum so that ordinary and sequential searches have the
same total budget; this stage answers the direct question: does squeezing lower the loss of the best re-fabricated device?
Output: data/rev_bounds2_<task>_sqf.npz  (best [6 params, r, re, dth, phi_d, L, NRMSE, nmax], check at N_c + 4 or + 8)."""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *

if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        task = TASKS[name]; fp = os.path.join(DATA, f'rev_bounds2_{name}_sqf.npz')
        if os.path.exists(fp): continue
        d = np.load(os.path.join(DATA, f'rev_bounds2_{name}.npz'), allow_pickle=True)
        bord = rb.best_row(d['ord_trace'])
        pd, th, L, S, n = rb.squeeze_on(rb.device(bord[:6]), task, bord[10])
        best = np.array([*bord[:6], *th, pd, *L])
        chk = rb.check(rb.device(bord[:6], pd), th, task, L[2]); chk0 = rb.check(rb.device(bord[:6]), (0, 0, 0), task, bord[12])
        rel = np.array([r2.evaluate_rel(rb.device(bord[:6]), (0, 0, 0), task), r2.evaluate_rel(rb.device(bord[:6], pd), th, task)])
        np.savez(fp, best=best, scans=S, evals=np.array(n), chk=chk, chk0=chk0, rel=rel)
        log(name, 'SQUEEZING ON FINAL ORDINARY', th.round(3), 'phi_d %.2f L %.4e -> %.4e (%.1f%%), checked %.4e -> %.4e (%.1f%%); relative objective %.4e -> %.4e' % (
            pd, bord[10], L[0], 100 * (1 - L[0] / bord[10]), chk0[-1][0], chk[-1][0], 100 * (1 - chk[-1][0] / chk0[-1][0]), rel[0][0], rel[1][0]))
