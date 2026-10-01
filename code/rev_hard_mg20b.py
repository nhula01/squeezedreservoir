"""Revision 2: second, independent ordinary search for Mackey-Glass at horizon 20 (the one case of rev_hard.py where a squeezed
re-fabricated device beats the best ordinary device found after 956 evaluations). Fresh Latin hypercube (different seed),
seeded with the two best ordinary devices found so far, 506 evaluations; then squeezing on its optimum.
Output: data/rev_hard_mg20_b.npz"""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *
from rev_hard import LADDER
if __name__ == '__main__':
    task = LADDER['mg20'][1](); h = np.load(os.path.join(DATA, 'rev_hard_mg20.npz'), allow_pickle=True); c = np.load(os.path.join(DATA, 'rev_hard_mg20_conv.npz'), allow_pickle=True)
    seeds = [rb.to_u(rb.best_row(h['ord_trace'])[:6]), rb.to_u(c['ord_best'][:6])]
    obj = rb.Obj(task, rb.B_TOT, False); rb.global_search(obj, 6, seeds, 777, rb.B_MAIN, rb.B_TOT)
    T = np.array(obj.trace); b = rb.best_row(T)
    pd, th, L, S, n = rb.squeeze_on(rb.device(b[:6]), task, b[10])
    c0 = rb.check(rb.device(b[:6]), (0, 0, 0), task, b[12]); c1 = rb.check(rb.device(b[:6], pd), th, task, L[2])
    np.savez(os.path.join(DATA, 'rev_hard_mg20_b.npz'), trace=T, ord_best=b, sq_best=np.array([*b[:6], *th, pd, *L]), chk0=c0, chk=c1)
    log('mg20 second search: ordinary %.4e (NRMSE %.3f), squeezed %.4e (%.1f%%); best squeezed so far %.4e' % (c0[-1][0], b[11], c1[-1][0], 100 * (1 - c1[-1][0] / c0[-1][0]), h['chk'][-1][0]))
