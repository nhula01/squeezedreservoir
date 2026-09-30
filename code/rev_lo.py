"""Squeezed input noise at the detector. The external squeezed vacuum enters and leaves through the measured port, so the
white homodyne noise of the cavity output depends on the LO angle: s(phi) = (eta_c V_in(phi) + 1 - eta_c)/(eta_c kappa tau).
For the tuned base and its squeezed optimum of every task, and for several LO-angle choices of the two cavity settings,
the clean loss of the resulting features and the measured loss against N_rep (12 noise draws) are computed:
  base : (0, pi/2) [Q and P], (0, 0) [Q only], (pi/2, pi/2) [P only]    (vacuum input noise at every angle)
  sq   : (0, pi/2) with vacuum noise [model of the first revision], (0, pi/2) with squeezed input noise,
         (phi_sq, phi_sq) [both settings in the squeezed quadrature], (phi_sq, phi_sq + pi/2).
The same study is repeated with the non-ideal bath (eta_inj = 0.7, sig_phi = 0.1)."""
from rev_common import *
NREP = np.array([1e4, 1e5, 1e6, 1e7, 1e8])

def study(dev, th, task, lo, vac, rng, eta_inj=1.0, sig_phi=0.0, ndraw=12):
    res = ResLO(dev, th, eta_inj, sig_phi, lo=lo); F, E, _ = res.run_moments(task.f)
    Vin = (1.0, 1.0) if vac else res.Vin
    cov = single_run_cov_sq(dev, F, E, Vin); L0 = evaluate(dev, th, task, feats=F)[:2]; rows = []
    for N in NREP:
        a = np.array([evaluate(dev, th, task, feats=add_readout_noise(dev, F, cov, N, rng))[:2] for _ in range(ndraw)])
        rows.append([N, *a.mean(0), *a.std(0)])
    return np.array(L0), np.array(rows), np.array(Vin)

if __name__ == '__main__':
    fp = os.path.join(DATA, 'rev_lo.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    rng = np.random.default_rng(11)
    for tag, kw in (('', {}), ('_ni', dict(eta_inj=0.7, sig_phi=0.1))):
        for n, task in TASKS.items():
            if f'{n}{tag}_done' in out: continue
            dev, th = squeezed_point2(n)
            N_, M_ = bath_NM(th, dev, kw.get('eta_inj', 1.0), kw.get('sig_phi', 0.0)); ps = squeezed_angle(N_, M_)
            cfg = {'base_QP': (np.zeros(3), (0, np.pi / 2), True), 'base_QQ': (np.zeros(3), (0, 0), True), 'base_PP': (np.zeros(3), (np.pi / 2, np.pi / 2), True),
                   'sq_QPvac': (th, (0, np.pi / 2), True), 'sq_QP': (th, (0, np.pi / 2), False), 'sq_SS': (th, (ps, ps), False), 'sq_SA': (th, (ps, ps + np.pi / 2), False)}
            for k, (t, lo, vac) in cfg.items():
                L0, rows, Vin = study(dev, t, task, lo, vac, rng, **(kw if k.startswith('sq') else {}))
                out[f'{n}{tag}_{k}_L0'] = L0; out[f'{n}{tag}_{k}'] = rows; out[f'{n}{tag}_{k}_Vin'] = Vin
            out[f'{n}{tag}_phisq'] = ps; out[f'{n}{tag}_done'] = 1
            bb = min((out[f'{n}{tag}_{k}'][:, 1] for k in ('base_QP', 'base_QQ', 'base_PP')), key=lambda v: v[2])
            msg = ' | '.join('%s clean %.3e: %s' % (k, out[f'{n}{tag}_{k}_L0'][0], ' '.join('%.0e:%+.0f%%' % (N, -100 * (1 - out[f'{n}{tag}_{k}'][i, 1] / np.min([out[f'{n}{tag}_{b}'][i, 1] for b in ('base_QP', 'base_QQ', 'base_PP')]))) for i, N in enumerate(NREP)))
                             for k in ('base_QP', 'base_QQ', 'sq_QPvac', 'sq_QP', 'sq_SS', 'sq_SA'))
            log(n + tag, 'phi_sq %.2f V(phi_sq) %.2f' % (ps, input_variance(ps, N_, M_)), msg)
            np.savez(fp, NREP=NREP, **out)
    log('done lo')
