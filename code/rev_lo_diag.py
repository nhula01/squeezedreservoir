"""Diagnostics for the squeezed-detector-noise study: (i) measured loss / clean loss with all readout noise and with the
emitter channel noiseless, for (Q,P) with vacuum and with squeezed input noise and for both settings in the squeezed
quadrature; (ii) a cavity-only readout (Q, P, Q^2, P^2) with LO pairs (0, pi/2), (phi_sq, phi_sq + pi/2), (phi_sq, phi_sq)."""
from rev_common import *
if __name__ == '__main__':
    rng = np.random.default_rng(4); out = {}; NN = np.array([1e4, 1e5, 1e6, 1e7])
    for n in ['nce', 'narma', 'lorenz', 'laser']:
        dev, th = squeezed_point2(n); task = TASKS[n]; N_, M_ = bath_NM(th, dev); ps = squeezed_angle(N_, M_)
        out[f'{n}_V'] = np.array([input_variance(0, N_, M_), input_variance(np.pi / 2, N_, M_), input_variance(ps, N_, M_), input_variance(ps + np.pi / 2, N_, M_)])
        cav = np.zeros((dev.Nv, 6), bool); cav[:, :4] = True; cav = cav.ravel()
        for lab, t, lo, vac in [('baseQP', np.zeros(3), (0, np.pi / 2), True), ('QPvac', th, (0, np.pi / 2), True), ('QP', th, (0, np.pi / 2), False),
                                ('SA', th, (ps, ps + np.pi / 2), False), ('SS', th, (ps, ps), False)]:
            r = ResLO(dev, t, lo=lo); F, E, _ = r.run_moments(task.f); cov = single_run_cov_sq(dev, F, E, (1, 1) if vac else r.Vin)
            covc = dict(cov); covc['vsx'] = 0 * cov['vsx']; covc['vsy'] = 0 * cov['vsy']
            L0 = evaluate(dev, t, task, feats=F)[0]; L0c = evaluate(dev, t, task, feats=F[:, cav])[0]; rows = []
            for N in NN:
                a = np.mean([evaluate(dev, t, task, feats=add_readout_noise(dev, F, cov, N, rng))[0] for _ in range(8)])
                b = np.mean([evaluate(dev, t, task, feats=add_readout_noise(dev, F, covc, N, rng))[0] for _ in range(8)])
                c = np.mean([evaluate(dev, t, task, feats=add_readout_noise(dev, F, cov, N, rng)[:, cav])[0] for _ in range(8)])
                rows.append([N, a, b, c])
            out[f'{n}_{lab}'] = np.array(rows); out[f'{n}_{lab}_L0'] = np.array([L0, L0c])
        log(n, 'V(Q,P,sq,anti)', out[f'{n}_V'].round(2))
    np.savez(os.path.join(DATA, 'rev_lo_diag.npz'), NN=NN, **out)
