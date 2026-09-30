"""Higher-cutoff rerun of rev_refab_sq.py for the tasks whose squeezed re-fabricated optimum sits near the photon budget
(Mackey-Glass, Lorenz). The N_c = 10 scans of rev_refab_sq.py are reused: their eight best admissible points are
re-evaluated at N_c = 14, convergent-gradient descent (60 evaluations) runs at N_c = 14 from the best of them, and the
optimum is checked at N_c = 18."""
from rev_common import *
from rev_refab import PN
from rev_base2 import Bud, R, D8
NC = 14

if __name__ == '__main__':
    which = sys.argv[1:] or ['mg', 'lorenz']
    rf = np.load(os.path.join(DATA, 'rev_refab.npz'), allow_pickle=True)
    sq = np.load(os.path.join(DATA, 'rev_refab_sq.npz'), allow_pickle=True)
    fp = os.path.join(DATA, 'rev_refab_sq14.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    for name in which:
        if f'{name}_best' in out: continue
        task = TASKS[name]; best = rf[f'{name}_best']
        dev = replace(DEV, Nc=NC, **dict(zip(PN, best[:6])))
        Lrf = evaluate(dev, (0, 0, 0), task); log(name, 'refab device at Nc %d: L %.4e (Nc10 %.4e) nmax %.2f' % (NC, Lrf[0], best[6], Lrf[2]))
        cands = []
        for pd in (0.0, np.pi / 2):
            S = sq[f'{name}_scan_{int(round(pd*2/np.pi))}']
            for idx in np.argsort(S[..., 0], axis=None)[:4]:
                i, j, k = np.unravel_index(idx, S.shape[:3])
                if np.isfinite(S[i, j, k, 0]): cands.append((pd, np.array([R[i], R[j], D8[k]])))
        re14 = []
        for pd, th in cands:
            L, nr, nm = evaluate(replace(dev, phi_d=pd), th, task); re14.append([pd, *th, L, nr, nm])
            log(name, 'scan point phi_d %.2f' % pd, th.round(3), 'Nc14 %.4e nmax %.2f' % (L, nm))
        re14 = np.array(re14); ok = re14[:, 6] <= NMAX; i0 = np.where(ok)[0][re14[ok, 4].argmin()]
        pd, th0 = re14[i0, 0], re14[i0, 1:4]; out[f'{name}_rescan14'] = re14
        f = Bud(replace(dev, phi_d=pd), task, 60)
        try: gradient_descent_conv(f, th0, 60)
        except StopIteration: pass
        T = np.array(f.trace); T = T[T[:, 5] <= NMAX]; jb = T[:, 3].argmin(); th = T[jb, :3]
        if T[jb, 3] > re14[i0, 4]: th = th0
        conv = [evaluate(replace(dev, phi_d=pd, Nc=Nc), th, task) for Nc in (NC, NC + 4)]
        Lrf18 = evaluate(replace(dev, Nc=NC + 4), (0, 0, 0), task)
        out[f'{name}_Lrf'] = np.array([Lrf, Lrf18]); out[f'{name}_best'] = np.array([pd, *th]); out[f'{name}_Lsq'] = np.array(conv); out[f'{name}_gdtrace'] = T
        log(name, 'squeezing on refab at Nc14: phi_d %.2f theta' % pd, th.round(3),
            'L %.4e -> %.4e (%.1f%%), Nc18 %.4e vs %.4e (%.1f%%), nmax %.2f, NRMSE %.3f -> %.3f' % (
                Lrf[0], conv[0][0], 100 * (1 - conv[0][0] / Lrf[0]), conv[1][0], Lrf18[0], 100 * (1 - conv[1][0] / Lrf18[0]), conv[0][2], Lrf[1], conv[0][1]))
        np.savez(fp, **out)
    log('done refab_sq14')
