"""Revision 2: the tuned-drive headline gains (Sec. 2.4, rev_base2.py) on fresh random realizations.
For NARMA10 and channel equalization, eight new input/target realizations (seeds 1-8; seed 0 is the one of the paper) are
generated. For each: (a) transfer: the tuned base and squeezed settings of the paper evaluated unchanged; (b) re-optimized:
the drive (eps, common detuning delta) re-tuned at zero squeezing by Nelder-Mead from the paper's base (30 evaluations),
then the squeezing controls re-optimized by convergent-gradient descent (60 evaluations) from the paper's squeezed optimum
on that re-tuned base. Output: data/rev_seeds.npz, rows [seed, L_base, L_sq, gain, NRMSE_base, NRMSE_sq] per mode."""
from rev_common import *
from sqz import make_narma, make_nce
from rev_base2 import Bud
from scipy.optimize import minimize
NC = 10; SEEDS = range(1, 9)
MAKE = {'narma': lambda s: make_narma(seed=s), 'nce': lambda s: make_nce(seed=s)}

def run(name, seed):
    task = MAKE[name](seed); b0 = base_device2(name, NC); _, th0 = squeezed_point2(name, NC)
    Lb = evaluate(b0, (0, 0, 0), task); Ls = evaluate(b0, th0, task)
    trans = [seed, Lb[0], Ls[0], 1 - Ls[0] / Lb[0], Lb[1], Ls[1]]
    tr = []
    def fb(x):
        e = float(np.clip(x[0], 0.05, 1.6)); dl = float(np.clip(x[1], -0.7, 1.0))
        dv = replace(b0, eps=e, omega=1 + dl, omega_q=1 + dl); L, nr, nm = evaluate(dv, (0, 0, 0), task); tr.append([e, dl, L, nr, nm])
        return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))
    x0 = np.array([b0.eps, b0.omega - 1]); minimize(fb, x0, method='Nelder-Mead', options=dict(maxfev=30, initial_simplex=[x0, x0 + [0.1 * x0[0], 0], x0 + [0, 0.1]]))
    tr = np.array(tr); ok = tr[:, 4] <= NMAX; e, dl, L, nr, nm = tr[ok][tr[ok, 2].argmin()]
    base = replace(b0, eps=e, omega=1 + dl, omega_q=1 + dl)
    f = Bud(base, task, 60)
    try: gradient_descent_conv(f, th0, 60)
    except StopIteration: pass
    T = np.array(f.trace); T = T[T[:, 5] <= NMAX]; j = T[:, 3].argmin()
    reopt = [seed, L, T[j, 3], 1 - T[j, 3] / L, nr, T[j, 4]]
    return np.array(trans), np.array(reopt), np.array([e, dl, *T[j, :3]])

if __name__ == '__main__':
    which = sys.argv[1:] or ['narma', 'nce']
    fp = os.path.join(DATA, 'rev_seeds.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    for name in which:
        for s in SEEDS:
            if f'{name}_{s}_reopt' in out: continue
            a, b, p = run(name, s); out[f'{name}_{s}_trans'] = a; out[f'{name}_{s}_reopt'] = b; out[f'{name}_{s}_set'] = p
            log(name, 'seed', s, 'transfer gain %.1f%%, re-optimized gain %.1f%% (L %.4e -> %.4e)' % (100 * a[3], 100 * b[3], b[1], b[2]))
            np.savez(fp, **out)
        R = np.array([out[f'{name}_{s}_reopt'] for s in SEEDS]); A = np.array([out[f'{name}_{s}_trans'] for s in SEEDS])
        log(name, 'SUMMARY re-optimized gain %.1f +- %.1f %% (min %.1f); transfer %.1f +- %.1f %%' % (
            100 * R[:, 3].mean(), 100 * R[:, 3].std(ddof=1), 100 * R[:, 3].min(), 100 * A[:, 3].mean(), 100 * A[:, 3].std(ddof=1)))
    log('done seeds')
