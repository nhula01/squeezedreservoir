"""Concern 2b: non-idealities at the squeezed optimum of every task (revised base device: drive amplitude and frequency optimized; N_c = 10).
(1) injection efficiency eta_inj of the external squeezed vacuum, at the commanded optimum and after re-optimization
    (convergent-gradient descent, 30 evaluations, same control box);
(2) fast phase jitter sig_phi of the squeezed vacuum relative to the parametric pump;
(3) slow drift of the controls and of the drive between evaluations (two levels), for the squeezed and the base device;
(4) everything at once, with the readout model at N_rep = 1e6 passes per setting, squeezed (re-optimized for eta 0.7)
    against the base device under the same drift and readout."""
from rev_common import *
NC = 10; ETAS = [1.0, 0.9, 0.7, 0.5]; SIGS = [0.05, 0.1, 0.2, 0.3]
DRIFTS = {'moderate': dict(r=0.01, re=0.01, eps=0.01, dth=0.02, phid=0.02), 'large': dict(r=0.03, re=0.03, eps=0.03, dth=0.05, phid=0.05)}
NDRAW = 10
if __name__ == '__main__':
    which = sys.argv[1:] or list(TASKS)
    fp = os.path.join(DATA, 'rev_nonideal.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    for name in which:
        if f'{name}_comb' in out: continue
        task = TASKS[name]; dsq, th = squeezed_point2(name, NC); db = dsq
        Lb = evaluate(db, (0, 0, 0), task)[0]; Ls = evaluate(dsq, th, task)[0]; out[f'{name}_Lb'] = Lb; out[f'{name}_Ls'] = Ls
        out[f'{name}_eta'] = np.array([ExpLoss(dsq, task, eta_inj=e)(th) for e in ETAS])
        reopt = {}
        for e in (0.7,):
            f = ExpLoss(dsq, task, eta_inj=e, budget=20, project=project)
            gradient_descent_conv(f, th, 20); tr = np.array(f.trace); ib = tr[:, 4].argmin(); reopt[e] = tr[ib, 1:4]
            out[f'{name}_reopt_{int(e*10)}'] = tr[ib]
        out[f'{name}_sig'] = np.array([ExpLoss(dsq, task, sig_phi=s)(th) for s in SIGS])
        for dn, dr in DRIFTS.items():
            rng = np.random.default_rng(3)
            fs = ExpLoss(dsq, task, drift=dr, rng=rng); fb = ExpLoss(db, task, drift=dr, rng=rng)
            out[f'{name}_drift_{dn}'] = np.array([[fs(th), fb(np.zeros(3))] for _ in range(NDRAW)])
        rng = np.random.default_rng(4); th7 = reopt[0.7]
        fs = ExpLoss(dsq, task, Nrep=1e6, eta_inj=0.7, sig_phi=0.1, drift=DRIFTS['moderate'], rng=rng)
        fb = ExpLoss(db, task, Nrep=1e6, drift=DRIFTS['moderate'], rng=rng)
        comb = np.array([[fs(th7), fb(np.zeros(3))] for _ in range(NDRAW)])
        # the same with the readout only (ideal bath, no drift), for the attribution
        rng = np.random.default_rng(5)
        fs2 = ExpLoss(dsq, task, Nrep=1e6, rng=rng); fb2 = ExpLoss(db, task, Nrep=1e6, rng=rng)
        ro = np.array([[fs2(th), fb2(np.zeros(3))] for _ in range(NDRAW)])
        out[f'{name}_comb'] = comb; out[f'{name}_ro'] = ro
        log(name, 'Lb %.4e Ls %.4e (gain %.0f%%)' % (Lb, Ls, 100 * (1 - Ls / Lb)),
            'eta', np.round(100 * (1 - out[f'{name}_eta'] / Lb)), 'reopt', {e: round(100 * (1 - out[f'{name}_reopt_{int(e*10)}'][4] / Lb)) for e in reopt},
            'sig', np.round(100 * (1 - out[f'{name}_sig'] / Lb)),
            'drift', {dn: round(100 * (1 - out[f'{name}_drift_{dn}'][:, 0].mean() / out[f'{name}_drift_{dn}'][:, 1].mean())) for dn in DRIFTS},
            'readout1e6 %.0f%%' % (100 * (1 - ro[:, 0].mean() / ro[:, 1].mean())), 'combined %.0f%%' % (100 * (1 - comb[:, 0].mean() / comb[:, 1].mean())))
        np.savez(fp, **out)
    log('done nonideal')
