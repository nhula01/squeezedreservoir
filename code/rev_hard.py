"""Revision 2: does squeezing add more to a re-fabricated device when the task is harder?
Difficulty ladders within three task families (level 0 is the task of the paper):
  Lorenz-63 prediction horizon 2 -> 5 -> 10 steps; Mackey-Glass horizon 10 -> 20 -> 40 steps;
  channel equalization target d(n-k) with delay k = 2 -> 4 -> 6.
For every new level: wide-box re-fabrication under the measurement-aware objective (rev_bounds2.py: 150 Latin-hypercube points
+ the level-0 wide-box optimum + the original-box optimum, three Nelder-Mead starts and continuation, 506 evaluations), then the
squeezing search of Sec. 2.4 on the converged optimum (206 evaluations), both checked at N_c + 4 (+ 8 near the photon budget).
Also the ten-delay linear regression on the same split, as a task-difficulty reference.
Usage: python3 rev_hard.py lorenz5 mg20 ...  -> data/rev_hard_<name>.npz"""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *
from sqz import make_lorenz, make_mg, Task

def make_nce_delay(k, N_total=1000, N_fade=10, N_train=690, seed=0, snr_db=24):
    """Channel equalization of Jaeger & Haas with target d(n-k) aligned with u(n)."""
    rng = np.random.default_rng(seed); N = N_total
    d = rng.choice([-3.0, -1.0, 1.0, 3.0], N + 20); q = np.zeros(N + 20)
    for n in range(7, N + 18):
        q[n] = (0.08 * d[n + 2] - 0.12 * d[n + 1] + d[n] + 0.18 * d[n - 1] - 0.1 * d[n - 2] + 0.091 * d[n - 3]
                - 0.05 * d[n - 4] + 0.04 * d[n - 5] + 0.03 * d[n - 6] + 0.01 * d[n - 7])
    u = q + 0.036 * q ** 2 - 0.011 * q ** 3
    u = u + rng.standard_normal(len(u)) * np.std(u) * 10 ** (-snr_db / 20)
    y = np.roll(d, k)
    return Task(f'nce{k}', u[10:N + 10], y[10:N + 10], N_fade, N_train, 0)

LADDER = {'lorenz5': ('lorenz', lambda: make_lorenz(horizon=5)), 'lorenz10': ('lorenz', lambda: make_lorenz(horizon=10)),
          'mg20': ('mg', lambda: make_mg(horizon=20)), 'mg40': ('mg', lambda: make_mg(horizon=40)),
          'nce4': ('nce', lambda: make_nce_delay(4)), 'nce6': ('nce', lambda: make_nce_delay(6))}

def ols10(task):
    """Ridge-free linear regression on the current and nine delayed inputs (same split), held-out NRMSE."""
    u = task.u; X = np.stack([np.roll(u, k) for k in range(10)] + [np.ones_like(u)], 1); tr, te = task.split()
    w = np.linalg.lstsq(X[tr], task.y[tr], rcond=None)[0]; p = X[te] @ w
    return np.sqrt(np.mean((task.y[te] - p) ** 2)) / np.std(task.y[te])

if __name__ == '__main__':
    import sys
    rf = np.load(os.path.join(DATA, 'rev_refab.npz'), allow_pickle=True)
    for name in sys.argv[1:]:
        base, mk = LADDER[name]; task = mk(); fp = os.path.join(DATA, f'rev_hard_{name}.npz')
        out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
        save = lambda: np.savez(fp, **out)
        out['ols10'] = np.array(ols10(task))
        if 'ord_trace' not in out:
            p0 = rb.best_row(np.load(os.path.join(DATA, f'rev_bounds2_{base}.npz'), allow_pickle=True)['ord_trace'])[:6]
            seeds = [rb.to_u(np.clip(p0, rb.LOW, rb.HIW)), rb.to_u(np.clip(rf[f'{base}_best'][:6], rb.LOW, rb.HIW))]
            obj = rb.Obj(task, rb.B_TOT, False); rb.global_search(obj, 6, seeds, 500 + hash(name) % 100, rb.B_MAIN, rb.B_TOT)
            out['ord_trace'] = np.array(obj.trace); save()
        bord = rb.best_row(out['ord_trace']); out['ord_best'] = bord
        log(name, 'ORDINARY', dict(zip(rb.PN, bord[:6].round(3))), 'L %.4e NRMSE %.3f nmax %.2f (OLS10 NRMSE %.3f)' % (bord[10], bord[11], bord[12], out['ols10']))
        if 'sq_best' not in out:
            pd, th, L, S, n = rb.squeeze_on(rb.device(bord[:6]), task, bord[10])
            out['sq_best'] = np.array([*bord[:6], *th, pd, *L]); out['sq_scans'] = S
            out['chk0'] = rb.check(rb.device(bord[:6]), (0, 0, 0), task, bord[12]); out['chk'] = rb.check(rb.device(bord[:6], pd), th, task, L[2])
            save()
        c0, c1 = out['chk0'][-1][0], out['chk'][-1][0]; sb = out['sq_best']
        log(name, 'SQUEEZING on re-fabricated', sb[6:9].round(3), 'phi_d %.2f: L %.4e -> %.4e, checked %.4e -> %.4e (%.2f%%), NRMSE %.3f -> %.3f' % (
            sb[9], bord[10], sb[10], c0, c1, 100 * (1 - c1 / c0), bord[11], sb[11]))
    log('done hard', sys.argv[1:])
