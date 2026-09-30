"""Squeezing on top of the re-fabricated device. The re-fabrication of rev_refab.py optimizes
(g, kappa, gamma, omega_a, omega_q, eps) at zero squeezing; here the squeezing controls (r, r_e, dtheta) are searched
from that re-fabricated optimum with exactly the protocol used from the tuned base (rev_base2.py): 3x3x8 scans of the
control box for phi_d = 0 and pi/2, convergent-gradient descent (60 evaluations) from the best admissible scan point,
photon budget 2.5, optimum checked at N_c + 4. This answers whether squeezing is a complement to design (it still lowers
the loss of the best unsqueezed device) or only a substitute for it."""
from rev_common import *
from rev_refab import PN
from rev_base2 import Bud, NC, R, D8

if __name__ == '__main__':
    which = sys.argv[1:] or list(TASKS)
    rf = np.load(os.path.join(DATA, 'rev_refab.npz'), allow_pickle=True)
    fp = os.path.join(DATA, 'rev_refab_sq.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    for name in which:
        if f'{name}_best' in out: continue
        task = TASKS[name]; best = rf[f'{name}_best']
        dev = replace(DEV, Nc=NC, **dict(zip(PN, best[:6])))
        Lrf = float(best[6]); nm0 = float(best[8])
        log(name, 'refab device', dict(zip(PN, best[:6].round(3))), 'L %.4e nmax %.2f' % (Lrf, nm0))
        scans = {}
        for pd in (0.0, np.pi / 2):
            S = np.zeros((3, 3, 8, 3))
            for i, r in enumerate(R):
                for j, re in enumerate(R):
                    for k, d in enumerate(D8): S[i, j, k] = evaluate(replace(dev, phi_d=pd), (r, re, d), task)
            S[..., 0] = np.where(S[..., 2] <= NMAX, S[..., 0], np.inf); scans[pd] = S; out[f'{name}_scan_{int(round(pd*2/np.pi))}'] = S
            log(name, 'scan phi_d %.2f best %.4e (%.0f%%)' % (pd, S[..., 0].min(), 100 * (1 - S[..., 0].min() / Lrf)))
        pd = min(scans, key=lambda p: scans[p][..., 0].min()); S = scans[pd]
        i, j, k = np.unravel_index(S[..., 0].argmin(), S.shape[:3]); th0 = np.array([R[i], R[j], D8[k]])
        f = Bud(replace(dev, phi_d=pd), task, 60)
        try: gradient_descent_conv(f, th0, 60)
        except StopIteration: pass
        T = np.array(f.trace); T = T[T[:, 5] <= NMAX]; jb = T[:, 3].argmin(); th = T[jb, :3]
        if T[jb, 3] > S[..., 0].min(): th = th0                      # descent never improved on the scan point
        conv = [evaluate(replace(dev, phi_d=pd, Nc=Nc), th, task) for Nc in (NC, NC + 4)]
        out[f'{name}_refab'] = best; out[f'{name}_best'] = np.array([pd, *th]); out[f'{name}_Lsq'] = np.array(conv); out[f'{name}_gdtrace'] = T
        out[f'{name}_evals'] = np.array(2 * S.size // 3 + f.n + 2)
        log(name, 'squeezing on refab: phi_d %.2f theta' % pd, th.round(3),
            'L %.4e -> %.4e (%.1f%%), Nc+4 %.4e (%.1f%%), nmax %.2f, NRMSE %.3f -> %.3f' % (
                Lrf, conv[0][0], 100 * (1 - conv[0][0] / Lrf), conv[1][0], 100 * (1 - conv[1][0] / float(rf[f'{name}_bestconv'][0])),
                conv[0][2], best[7], conv[0][1]))
        np.savez(fp, **out)
    log('done refab_sq')
