"""Revision 2: validate the bin readout model of Sec. 2.6 / Methods against quantum trajectories of the actual
homodyne currents, at the channel-equalization operating points of the cost study (tuned base, r = 0, vacuum input; and
its squeezed optimum, r_e = 0.5).

Trajectory model. The ideal squeezed bath of sqz.build is D[L_c], L_c = sqrt(kappa)(mu a + nu a^dag), mu = cosh r_e,
nu = s e^{i dth} sinh r_e. It is generated exactly by a vacuum field b0 through b_in = mu b0 - nu b0^dag, so that the
physical output b_out = b_in + sqrt(kappa) a equals mu b0_out - nu b0_out^dag with b0_out = b0 + L_c. A homodyne
measurement of the physical output quadrature x_Phi(b_out) is therefore the measurement of x_Phi'(b0_out) scaled by
A = |mu e^{-i Phi} - nu^* e^{i Phi}| = sqrt(V_in(Phi)), with e^{-i Phi'} = (mu e^{-i Phi} - nu^* e^{i Phi})/A, and the
operator identity A x_Phi'(L_c) = sqrt(kappa) x_Phi(a) holds. Detection efficiency eta on the physical field is an
efficiency eta' = eta V_in/(eta V_in + 1 - eta) on b0_out. Both damping channels (cavity, emitter) are unravelled by
ideal homodyne detection (stochastic Schroedinger equation, Ito/Euler, dt = 0.005 with a dt/2 check); inefficiency is
added to the recorded currents as independent white noise, which leaves the distribution of the physical record exact.
Every record after t0 depends only on rho(t0), so trajectories start from pure states sampled from the eigen-
decomposition of the unconditional state at the start of a 12-symbol training segment.

Estimators per bin tau = T_in/N_v (as an experiment would form them from one pass):
  Q^ = A J_c/(sqrt(kappa) tau),   Q2^ = Q^2 - s_c(Phi),   sx^ = J_e/(sqrt(gamma) tau),
  s_c(Phi) = (eta_c V_in + 1 - eta_c)/(eta_c kappa tau)   (the bias-free Q^2 estimator).
Compared with the bin model (expt.single_run_cov, vacuum s_c as in the main text; expt.single_run_cov_sq, input-noise
s_c(Phi) as in Supplementary Note 15): variances of Q^, Q2^, sx^, the (Q^, Q2^) covariance, and the means (the bin
average of the current against the end-of-bin feature of the model). Then the loss noise at N_rep = 1e5..1e7 is
recomputed with the model covariance rescaled by the trajectory/model ratio of each feature class.
Output: data/rev_sme.npz"""
from rev_common import *
from sqz import _ops, build
from expt import ResX, single_run_cov, single_run_cov_sq, add_readout_noise, input_variance, bath_NM
import scipy.linalg as sla
NC = 10; ETA_C, ETA_E = 0.8, 0.5; T0, NSYM = 300, 12; B = 8000

def operators(dev, th):
    r, re, dth = th
    al, sig = _ops(dev.Nc); ald = al.conj().T
    a = np.cosh(r) * al + dev.s * np.sinh(r) * ald
    H0 = dev.omega / np.cosh(2 * r) * (ald @ al) + dev.omega_q * (sig.conj().T @ sig) + dev.g * (a.conj().T @ sig + a @ sig.conj().T)
    X = 1j * dev.eps * (a * np.exp(-1j * dev.phi_d) - a.conj().T * np.exp(1j * dev.phi_d))
    mu, nu = np.cosh(re), dev.s * np.exp(1j * dth) * np.sinh(re)
    Lc = np.sqrt(dev.kappa) * (mu * a + nu * a.conj().T); Le = np.sqrt(dev.gamma) * sig
    # consistency with the simulator's Lindblad operator
    L0 = build(dev, th)[0]
    from sqz import _comm, _dissipator
    assert np.abs(_comm(H0) + _dissipator(Lc) + _dissipator(Le) - L0).max() < 1e-10
    return dict(a=a, sig=sig, H0=H0, X=X, Lc=Lc, Le=Le, mu=mu, nu=nu)

def trajectories(dev, th, Phi, rho0, f_seg, nb, dt, rng):
    """Returns per-bin records (nb, B) of Q^, sx^ (setting Phi = 0: Q, sigma_x; Phi = pi/2: P, sigma_y) and A, s_c."""
    op = operators(dev, th); D = op['a'].shape[0]; tau = dev.T_in / dev.Nv; m = int(round(tau / dt))
    z = op['mu'] * np.exp(-1j * Phi) - np.conj(op['nu']) * np.exp(1j * Phi); A = abs(z); Vin = A ** 2
    etap = ETA_C * Vin / (ETA_C * Vin + 1 - ETA_C)
    Xc = op['Lc'] * (z / A); Xe = op['Le'] * np.exp(-1j * Phi)
    LdL = op['Lc'].conj().T @ op['Lc'] + op['Le'].conj().T @ op['Le']
    w, V = np.linalg.eigh(rho0); w = np.clip(w, 0, None); w /= w.sum()
    idx = rng.choice(D, size=B, p=w); psi = V[:, idx].T.copy()                      # (B, D)
    XcT, XeT = Xc.T.copy(), Xe.T.copy()
    Jc = np.zeros((nb, B)); Je = np.zeros((nb, B)); sq = np.sqrt(dt)
    for t, fv in enumerate(f_seg):
        K = (-1j * (op['H0'] + fv * op['X']) - 0.5 * LdL).T.copy()
        for v in range(dev.Nv):
            b = t * dev.Nv + v
            for _ in range(m):
                xc = psi @ XcT; xe = psi @ XeT
                mc = 2 * np.real(np.einsum('bi,bi->b', psi.conj(), xc)); me = 2 * np.real(np.einsum('bi,bi->b', psi.conj(), xe))
                dWc = sq * rng.standard_normal(B); dWe = sq * rng.standard_normal(B)
                Jc[b] += mc * dt + dWc; Je[b] += me * dt + dWe
                psi = psi + (psi @ K) * dt + (0.5 * mc[:, None] * xc + 0.5 * me[:, None] * xe - 0.125 * (mc ** 2 + me ** 2)[:, None] * psi) * dt \
                      + (xc - 0.5 * mc[:, None] * psi) * dWc[:, None] + (xe - 0.5 * me[:, None] * psi) * dWe[:, None]
                psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    Jc += np.sqrt((1 - etap) / etap) * np.sqrt(tau) * rng.standard_normal(Jc.shape)   # detector inefficiency
    Je += np.sqrt((1 - ETA_E) / ETA_E) * np.sqrt(tau) * rng.standard_normal(Je.shape)
    Qh = A * Jc / (np.sqrt(dev.kappa) * tau); sh = Je / (np.sqrt(dev.gamma) * tau)
    sc = (ETA_C * Vin + 1 - ETA_C) / (ETA_C * dev.kappa * tau)
    return Qh, sh, A, sc

def state_at(dev, th, f, t0):
    res = ResX(dev, th); _, _, states = Reservoir.run(res, f[:t0], return_states=True)
    return res.to_matrix(states[-1])

def binavg_features(dev, th, f, m=10):
    """Features averaged over each bin (midpoint rule, m sub-steps), as a time-integrated current estimates them."""
    res = ResX(dev, th); sub = replace(dev, T_in=dev.T_in / m)
    from sqz import _cheb_coeffs
    Ps = np.stack([sla.expm((res.L0r + xk * res.L1r) * res.tau / m) for xk in res.x]).reshape(dev.K, -1)
    Ph = np.stack([sla.expm((res.L0r + xk * res.L1r) * res.tau / (2 * m)) for xk in res.x]).reshape(dev.K, -1)
    C = _cheb_coeffs(np.asarray(f, float), res.x, res.w); D2 = res.D ** 2; rho = res.rho0.copy()
    out = np.zeros((len(f), dev.Nv, NOBS))
    for t in range(len(f)):
        P = (C[t] @ Ps).reshape(D2, D2); Pm = (C[t] @ Ph).reshape(D2, D2)
        for v in range(dev.Nv):
            acc = np.zeros(NOBS)
            for _ in range(m):
                acc += res.obs @ (Pm @ rho); rho = P @ rho
            out[t, v] = acc / m
    return out.reshape(len(f), NOBS * dev.Nv)

if __name__ == '__main__':
    task = TASKS['nce']; f = task.f; rng = np.random.default_rng(11)
    dsq, thsq = squeezed_point2('nce', NC)
    pts = {'base': (dsq, np.zeros(3)), 'sq': (dsq, thsq)}
    fp = os.path.join(DATA, 'rev_sme.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    NREP = np.array([1e5, 1e6, 1e7])
    for k, (dv, th) in pts.items():
        if f'{k}_ratio' in out: continue
        F, E, _ = ResX(dv, th).run_moments(f)
        seg = slice(T0, T0 + NSYM); nb = NSYM * dv.Nv
        Fs = F[seg].reshape(nb, NOBS); cov_v = single_run_cov(dv, F[seg], E[seg], ETA_C, ETA_E)
        N, M = bath_NM(th, dv); Vin = (input_variance(0.0, N, M), input_variance(np.pi / 2, N, M))
        cov_s = single_run_cov_sq(dv, F[seg], E[seg], Vin, ETA_C, ETA_E)
        rho0 = state_at(dv, th, f, T0); Fb = binavg_features(dv, th, f[:T0 + NSYM])[seg].reshape(nb, NOBS)
        res = {}
        for Phi, (iq, iq2, ie, kq, kq2, kc, ke) in ((0.0, (0, 2, 4, 'vQ', 'vQ2', 'cQ', 'vsx')), (np.pi / 2, (1, 3, 5, 'vP', 'vP2', 'cP', 'vsy'))):
            for dt in ((0.005, 0.0025) if Phi == 0.0 else (0.005,)):
                t1 = time.time(); Qh, sh, A, sc = trajectories(dv, th, Phi, rho0, f[seg], nb, dt, rng); Q2h = Qh ** 2 - sc
                traj = dict(vQ=Qh.var(1), vQ2=Q2h.var(1), cQ=np.array([np.cov(Qh[i], Q2h[i])[0, 1] for i in range(nb)]), vs=sh.var(1),
                            mQ=Qh.mean(1), mQ2=Q2h.mean(1), ms=sh.mean(1))
                mv = {c: cov_v[kk].reshape(nb) for c, kk in (('vQ', kq), ('vQ2', kq2), ('cQ', kc), ('vs', ke))}
                ms = {c: cov_s[kk].reshape(nb) for c, kk in (('vQ', kq), ('vQ2', kq2), ('cQ', kc), ('vs', ke))}
                tag = '%s_%s_dt%g' % (k, 'Q' if Phi == 0 else 'P', dt)
                for c in ('vQ', 'vQ2', 'cQ', 'vs', 'mQ', 'mQ2', 'ms'): out[f'{tag}_traj_{c}'] = traj[c]
                for c in ('vQ', 'vQ2', 'cQ', 'vs'): out[f'{tag}_modv_{c}'] = mv[c]; out[f'{tag}_mods_{c}'] = ms[c]
                out[f'{tag}_feat'] = Fs[:, [iq, iq2, ie]]; out[f'{tag}_featbin'] = Fb[:, [iq, iq2, ie]]
                res[tag] = {c: np.median(traj[c] / mv[c]) for c in ('vQ', 'vQ2', 'vs')}
                res[tag].update({c + '_sq': np.median(traj[c] / ms[c]) for c in ('vQ', 'vQ2', 'vs')})
                log(k, tag, '%.0f s' % (time.time() - t1), 'traj/model (vacuum s_c):', {c: round(v, 3) for c, v in res[tag].items()},
                    'A %.3f sc %.2f' % (A, sc), 'mean err Q %.3f (bin-avg %.3f)' % (np.median(np.abs(traj['mQ'] - Fs[:, iq])), np.median(np.abs(traj['mQ'] - Fb[:, iq]))))
                np.savez(fp, **out)
        # loss noise with the model covariance and with the trajectory-calibrated covariance
        cov = single_run_cov(dv, F, E, ETA_C, ETA_E); cal = dict(cov)
        for kk, c, s in (('vQ', 'vQ', 'Q'), ('vQ2', 'vQ2', 'Q'), ('cQ', None, 'Q'), ('vsx', 'vs', 'Q'), ('vP', 'vQ', 'P'), ('vP2', 'vQ2', 'P'), ('cP', None, 'P'), ('vsy', 'vs', 'P')):
            tag = '%s_%s_dt0.005' % (k, s)
            if c is None:
                rq = np.median(out[f'{tag}_traj_cQ'] / out[f'{tag}_modv_cQ']); cal[kk] = cov[kk] * rq
            else:
                cal[kk] = cov[kk] * np.median(out[f'{tag}_traj_{c}'] / out[f'{tag}_modv_{c}'])
        L0 = evaluate(dv, th, task, feats=F)[0]; Lb = evaluate(dv, th, task, feats=binavg_features(dv, th, f))[0]
        rows = []
        for Nr in NREP:
            a = [evaluate(dv, th, task, feats=add_readout_noise(dv, F, cov, Nr, rng))[0] for _ in range(24)]
            b = [evaluate(dv, th, task, feats=add_readout_noise(dv, F, cal, Nr, rng))[0] for _ in range(24)]
            rows.append([Nr, np.mean(a), np.std(a), np.mean(b), np.std(b)])
        out[f'{k}_loss'] = np.array(rows); out[f'{k}_L0'] = np.array([L0, Lb]); out[f'{k}_ratio'] = np.array(1)
        log(k, 'clean L %.4e, bin-averaged features %.4e' % (L0, Lb), ' | '.join('N=%.0e model %.4e+-%.1e  traj-cal %.4e+-%.1e' % tuple(r) for r in rows))
        np.savez(fp, **out)
    log('done sme')
