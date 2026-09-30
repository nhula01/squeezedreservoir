"""Base device including the drive frequency. The drive (and, with squeezing on, the parametric pump at twice its
frequency) can be tuned after fabrication, which shifts the cavity and emitter detunings omega_a, omega_q together by
delta. The base device is therefore re-optimized over (eps, delta) at zero squeezing (the 27-point grid of rev_refab plus a
Nelder-Mead refinement), and the squeezing search of Section 2.5 is repeated from that base: 3x3x8 scans of (r, r_e, dth)
for phi_d = 0 and pi/2, then convergent-gradient descent (60 evaluations) from the best scan point, photon budget 2.5."""
from rev_common import *
from scipy.optimize import minimize
NC = 10; R = np.linspace(0, 0.5, 3); D8 = np.linspace(-np.pi, np.pi, 9)[:-1]

class Bud:
    def __init__(self, dev, task, budget=None): self.dev, self.task, self.budget, self.n, self.trace = dev, task, budget, 0, []
    def __call__(self, th):
        if self.budget is not None and self.n >= self.budget: raise StopIteration
        th = project(th); L, nr, nm = evaluate(self.dev, th, self.task); self.n += 1; self.trace.append([*th, L, nr, nm])
        return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))

if __name__ == '__main__':
    which = sys.argv[1:] or list(TASKS)
    rf = np.load(os.path.join(DATA, 'rev_refab.npz'), allow_pickle=True)
    fp = os.path.join(DATA, 'rev_base2.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    for name in which:
        if f'{name}_best' in out: continue
        task = TASKS[name]; b0 = base_device(name, NC)
        rows = rf[f'{name}_drive']; ok = rows[:, 4] <= NMAX; i0 = np.where(ok)[0][rows[ok, 2].argmin()]
        tr = []
        def fb(x):
            e, dl = x; e = float(np.clip(e, 0.05, 1.6)); dl = float(np.clip(dl, -0.7, 1.0))
            L, nr, nm = evaluate(replace(b0, eps=e, omega=1 + dl, omega_q=1 + dl), (0, 0, 0), task); tr.append([e, dl, L, nr, nm])
            return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))
        x0 = rows[i0, :2]; minimize(fb, x0, method='Nelder-Mead', options=dict(maxfev=30, initial_simplex=[x0, x0 + [0.1 * x0[0], 0], x0 + [0, 0.1]]))
        tr = np.array(tr); allr = np.vstack([rows[:, :5], tr]); ok = allr[:, 4] <= NMAX; ib = np.where(ok)[0][allr[ok, 2].argmin()]
        eps2, dl2 = allr[ib, :2]; Lbase2 = allr[ib, 2]
        out[f'{name}_basegrid'] = allr; out[f'{name}_base2'] = allr[ib]
        base2 = replace(b0, eps=eps2, omega=1 + dl2, omega_q=1 + dl2)
        log(name, 'base2 eps %.3f delta %.3f L %.4e (old base %.4e)' % (eps2, dl2, Lbase2, float(BASE[f'{name}_eps'][:, 1].min())))
        scans = {}
        for pd in (0.0, np.pi / 2):
            S = np.zeros((3, 3, 8, 3))
            for i, r in enumerate(R):
                for j, re in enumerate(R):
                    for k, d in enumerate(D8): S[i, j, k] = evaluate(replace(base2, phi_d=pd), (r, re, d), task)
            S[..., 0] = np.where(S[..., 2] <= NMAX, S[..., 0], np.inf); scans[pd] = S; out[f'{name}_scan_{int(round(pd*2/np.pi))}'] = S
        pd = min(scans, key=lambda p: scans[p][..., 0].min()); S = scans[pd]
        i, j, k = np.unravel_index(S[..., 0].argmin(), S.shape[:3]); th0 = np.array([R[i], R[j], D8[k]])
        f = Bud(replace(base2, phi_d=pd), task, 60); gradient_descent_conv(f, th0, 60)
        T = np.array(f.trace); T = T[T[:, 5] <= NMAX]; jb = T[:, 3].argmin(); th = T[jb, :3]
        conv = [evaluate(replace(base2, phi_d=pd, Nc=Nc), th, task) for Nc in (NC, NC + 4)]
        out[f'{name}_best'] = np.array([eps2, dl2, pd, *th]); out[f'{name}_Lsq2'] = np.array(conv); out[f'{name}_gdtrace'] = T
        log(name, 'squeezing from base2: phi_d %.2f theta' % pd, th.round(3), 'L %.4e -> %.4e (%.0f%%), Nc+4 %.4e' % (Lbase2, conv[0][0], 100 * (1 - conv[0][0] / Lbase2), conv[1][0]))
        np.savez(fp, **out)
    log('done base2')
