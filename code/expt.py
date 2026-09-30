"""Experimental model for the referee-driven revision (cost and non-idealities).

1. Non-ideal squeezed bath. The injected squeezed vacuum of source squeezing r_e passes an injection
   efficiency eta_inj (escape/mode matching/propagation loss) and carries fast phase jitter of rms sig_phi
   relative to the parametric pump. The cavity then sees a Gaussian bath with
        N = eta_inj sinh^2 r_e + n_th,     M = eta_inj cosh r_e sinh r_e e^{i theta} e^{-sig_phi^2/2},
   |M|^2 <= N(N+1), and the bath dissipator is
        kappa(N+1) D[a] + kappa N D[a^dag] + kappa M S[a^dag] + kappa M* S[a],   S[x]rho = x rho x - {x x, rho}/2.
   For eta_inj = 1, sig_phi = 0, n_th = 0 this is exactly D[L_a] of sqz.build (checked in selftest()).

2. Readout model. Features are estimated from repeated passes of the input sequence with time-resolved,
   non-demolition homodyne detection of the cavity output (rate kappa, efficiency eta_c) and of the emitter
   fluorescence (rate gamma, efficiency eta_e). One pass with the local-oscillator phases set to (Q, sigma_x)
   or to (P, sigma_y) yields one single-run estimate of every feature of that setting at every virtual time.
   With bin time tau = T_in/N_v and white detector noise of variance s = 1/(eta kappa tau) per bin,
        Var[Q^]   = Var(Q) + s_c,           Var[Q2^] = Var(Q^2) + 4<Q^2> s_c + 2 s_c^2,
        Cov[Q^,Q2^] = <Q^3> - <Q><Q^2> + 2<Q> s_c,
        Var[sx^]  = 1 - <sx>^2 + s_e,       s_c = 1/(eta_c kappa tau), s_e = 1/(eta_e gamma tau),
   and the same for P, sigma_y. Averages over N_rep passes per setting divide these by N_rep (two settings,
   so one loss evaluation costs 2 N_rep passes of the sequence).

3. Slow drift. Commanded controls and drive are realized with per-evaluation Gaussian errors
   (relative on r, r_e, eps; absolute on dtheta, phi_d).
"""
import numpy as np
import scipy.linalg as sla
from dataclasses import replace
from sqz import Device, _ops, _sop, _dissipator, _comm, _herm_basis, _cheb_nodes, _cheb_coeffs, Reservoir, evaluate, NOBS

def _S(x):
    I = np.eye(x.shape[0])
    return _sop(x, x) - 0.5 * _sop(x @ x, I) - 0.5 * _sop(I, x @ x)

def build_ext(dev, theta, eta_inj=1.0, sig_phi=0.0, nth=0.0):
    r, re, dth = theta
    al, sig = _ops(dev.Nc)
    ald, sigd = al.conj().T, sig.conj().T
    ch, sh = np.cosh(r), np.sinh(r)
    a = ch * al + dev.s * sh * ald
    ad = a.conj().T
    Om = dev.omega / np.cosh(2 * r)
    H0 = Om * (ald @ al) + dev.omega_q * (sigd @ sig) + dev.g * (ad @ sig + a @ sigd)
    X = 1j * dev.eps * (a * np.exp(-1j * dev.phi_d) - ad * np.exp(1j * dev.phi_d))
    eth = dev.s * np.exp(1j * dth)                      # lab-frame bath phase e^{i theta}, theta = dth + phi_s
    N = eta_inj * np.sinh(re) ** 2 + nth
    M = eta_inj * np.cosh(re) * np.sinh(re) * eth * np.exp(-0.5 * sig_phi ** 2)
    k = dev.kappa
    Lb = k * (N + 1) * _dissipator(a) + k * N * _dissipator(ad) + k * M * _S(ad) + k * np.conj(M) * _S(a)
    L0 = _comm(H0) + Lb + dev.gamma * _dissipator(sig)
    L1 = _comm(X)
    Q = a + ad; P = 1j * (ad - a)
    sx = sig + sigd; sy = 1j * (sigd - sig)
    ops = (Q, P, Q @ Q, P @ P, sx, sy, Q @ Q @ Q, Q @ Q @ Q @ Q, P @ P @ P, P @ P @ P @ P)
    obs = np.array([O.T.ravel() for O in ops])
    nvec = (ad @ a).T.ravel()
    return L0, L1, obs, nvec, a.shape[0]

class ResX(Reservoir):
    """Reservoir with the non-ideal bath and the extra moments needed by the readout-noise model."""
    def __init__(self, dev, theta, eta_inj=1.0, sig_phi=0.0, nth=0.0):
        self.dev, self.theta = dev, np.asarray(theta, float)
        L0, L1, obs, nvec, D = build_ext(dev, theta, eta_inj, sig_phi, nth)
        self.D = D
        B = _herm_basis(D)
        L0r = np.asarray(B.conj().T @ (L0 @ B)).real
        L1r = np.asarray(B.conj().T @ (L1 @ B)).real
        self.obs_all = np.asarray(obs @ B).real
        self.obs = self.obs_all[:NOBS]
        self.nvec = np.asarray(nvec @ B).real
        self.B = B
        tau = dev.T_in / dev.Nv
        self.x, self.w = _cheb_nodes(dev.K)
        self.Pflat = np.ascontiguousarray(np.stack([sla.expm((L0r + xk * L1r) * tau) for xk in self.x]).reshape(dev.K, -1))
        self.L0r, self.L1r, self.tau = L0r, L1r, tau
        self.rho0 = np.zeros(D * D); self.rho0[0] = 1.0

    def run_moments(self, f):
        """Returns features (T, 6 Nv) [same layout as Reservoir.run], extra moments (T, Nv, 4) = <Q^3>,<Q^4>,<P^3>,<P^4>, nmax."""
        dev, D2 = self.dev, self.D * self.D
        T = len(f); rho = self.rho0.copy()
        C = _cheb_coeffs(np.asarray(f, float), self.x, self.w)
        allm = np.empty((T, dev.Nv, self.obs_all.shape[0])); nmax = 0.0
        for t in range(T):
            P = (C[t] @ self.Pflat).reshape(D2, D2)
            for v in range(dev.Nv):
                rho = P @ rho
                allm[t, v] = self.obs_all @ rho
                nmax = max(nmax, self.nvec @ rho)
        feats = allm[:, :, :NOBS].reshape(T, NOBS * dev.Nv)
        return feats, allm[:, :, NOBS:], nmax

# ---------------------------------------------------------------- readout noise
def single_run_cov(dev, feats, extra, eta_c=0.8, eta_e=0.5):
    """Per-bin single-run variances/covariances of the feature estimators (see module docstring)."""
    T = feats.shape[0]; Nv = dev.Nv
    F = feats.reshape(T, Nv, NOBS); Q, P, Q2, P2, sx, sy = [F[:, :, i] for i in range(NOBS)]
    Q3, Q4, P3, P4 = [extra[:, :, i] for i in range(4)]
    tau = dev.T_in / Nv
    sc = 1.0 / (eta_c * dev.kappa * tau); se = 1.0 / (eta_e * dev.gamma * tau)
    out = dict(
        vQ=Q2 - Q ** 2 + sc, vP=P2 - P ** 2 + sc,
        vQ2=np.maximum(Q4 - Q2 ** 2, 0) + 4 * Q2 * sc + 2 * sc ** 2,
        vP2=np.maximum(P4 - P2 ** 2, 0) + 4 * P2 * sc + 2 * sc ** 2,
        cQ=Q3 - Q * Q2 + 2 * Q * sc, cP=P3 - P * P2 + 2 * P * sc,
        vsx=1 - sx ** 2 + se, vsy=1 - sy ** 2 + se, sc=sc, se=se)
    return out

def add_readout_noise(dev, feats, cov, Nrep, rng):
    """Feature estimates from Nrep passes per setting (CLT Gaussian with the joint (Q,Q^2) covariance)."""
    if Nrep is None or not np.isfinite(Nrep):
        return feats
    T = feats.shape[0]; Nv = dev.Nv
    F = feats.reshape(T, Nv, NOBS).copy()
    for (iq, iq2, v1, v2, c) in ((0, 2, 'vQ', 'vQ2', 'cQ'), (1, 3, 'vP', 'vP2', 'cP')):
        a, b, cc = cov[v1], cov[v2], cov[c]
        z1, z2 = rng.standard_normal((2, T, Nv))
        s1 = np.sqrt(a); rho_ = np.clip(cc / np.sqrt(a * b), -1, 1)
        F[:, :, iq] += s1 * z1 / np.sqrt(Nrep)
        F[:, :, iq2] += np.sqrt(b) * (rho_ * z1 + np.sqrt(1 - rho_ ** 2) * z2) / np.sqrt(Nrep)
    F[:, :, 4] += np.sqrt(cov['vsx']) * rng.standard_normal((T, Nv)) / np.sqrt(Nrep)
    F[:, :, 5] += np.sqrt(cov['vsy']) * rng.standard_normal((T, Nv)) / np.sqrt(Nrep)
    return F.reshape(T, NOBS * Nv)

def realized_sigma_rel(dev, task, feats, cov, Nrep):
    """Median (over features) of the readout noise std divided by the training-window std of that feature."""
    tr, _ = task.split(); T = feats.shape[0]; Nv = dev.Nv
    V = np.zeros((T, Nv, NOBS))
    for i, k in enumerate(('vQ', 'vP', 'vQ2', 'vP2', 'vsx', 'vsy')):
        V[:, :, i] = cov[k]
    V = V.reshape(T, NOBS * Nv)
    sd = feats[tr].std(0)
    rel = np.sqrt(V[tr].mean(0) / Nrep) / np.where(sd > 1e-12, sd, np.nan)
    return np.nanmedian(rel), rel

# ---------------------------------------------------------------- one experimental loss evaluation
def realize(dev, theta, drift, rng):
    """Apply slow per-evaluation drift: dict with keys r, re (relative), dth, phid (rad), eps (relative)."""
    if not drift:
        return dev, np.asarray(theta, float)
    r, re, dth = theta
    r = max(0.0, r * (1 + drift.get('r', 0) * rng.standard_normal()))
    re = max(0.0, re * (1 + drift.get('re', 0) * rng.standard_normal()))
    dth = dth + drift.get('dth', 0) * rng.standard_normal()
    dev = replace(dev, eps=dev.eps * (1 + drift.get('eps', 0) * rng.standard_normal()),
                  phi_d=dev.phi_d + drift.get('phid', 0) * rng.standard_normal())
    return dev, np.array([r, re, dth])

class ExpLoss:
    """Loss as an experiment would measure it: drift -> non-ideal reservoir -> Nrep passes per setting -> ridge.
    Counts evaluations and sequence passes; records (n, theta, noisy L, clean-feature L at realized point)."""
    def __init__(self, dev, task, Nrep=None, eta_inj=1.0, sig_phi=0.0, drift=None, eta_c=0.8, eta_e=0.5, rng=None,
                 budget=None, project=None):
        self.dev, self.task, self.Nrep = dev, task, Nrep
        self.eta_inj, self.sig_phi, self.drift = eta_inj, sig_phi, drift
        self.eta_c, self.eta_e = eta_c, eta_e
        self.rng = rng or np.random.default_rng()
        self.budget, self.project = budget, project
        self.n = 0; self.passes = 0; self.trace = []

    def __call__(self, theta):
        if self.budget is not None and self.n >= self.budget:
            raise StopIteration
        th = self.project(theta) if self.project is not None else np.asarray(theta, float)
        dv, thr = realize(self.dev, th, self.drift, self.rng)
        res = ResX(dv, thr, self.eta_inj, self.sig_phi)
        feats, extra, nmax = res.run_moments(self.task.f)
        Lclean = evaluate(dv, thr, self.task, feats=feats)[0]
        if self.Nrep is not None:
            cov = single_run_cov(dv, feats, extra, self.eta_c, self.eta_e)
            feats = add_readout_noise(dv, feats, cov, self.Nrep, self.rng)
            self.passes += 2 * self.Nrep
        L, nr, _ = evaluate(dv, thr, self.task, feats=feats)
        self.n += 1
        self.trace.append((self.n, *th, L, Lclean, nmax, nr))
        return L

def selftest():
    from common import DEV, MG
    for th in [(0.2, 0.3, 0.7), (0.0, 0.4, -1.2), (0.5, 0.1, 2.0)]:
        for s in (1.0, -1.0):
            dv = replace(DEV, s=s, phi_d=0.3)
            from sqz import build
            A = build(dv, th)[0]; B = build_ext(dv, th)[0]
            assert np.abs(A - B).max() < 1e-12, (th, s, np.abs(A - B).max())
    f = MG.f[:200]
    a = Reservoir(DEV, (0.1, 0.3, 0.5)).run(f)[0]; b = ResX(DEV, (0.1, 0.3, 0.5)).run_moments(f)[0]
    assert np.abs(a - b).max() < 1e-10
    # physicality: the lossy bath keeps the state positive
    res = ResX(DEV, (0.0, 0.5, 0.0), eta_inj=0.5, sig_phi=0.3)
    _, _, nm = res.run_moments(f)
    print('selftest ok; lossy-bath nmax', nm)

if __name__ == '__main__':
    selftest()

# ---------------------------------------------------------------- noise-matched readout
def evaluate_nm(dev, task, feats, noise_var=None, sigma_floor=None):
    """Ridge readout with a per-feature, noise-matched penalty. Features are standardized on the training window;
    feature j gets the penalty N_tr * max(v_j, sigma_floor^2), v_j its readout-noise variance in standardized units,
    so the rule of the main text (uniform N_tr sigma_rel^2) is recovered when the readout noise is below sigma_rel.
    Returns training loss, held-out NRMSE."""
    from sqz import ridge_fit
    sigma_floor = dev.sigma_rel if sigma_floor is None else sigma_floor
    tr, te = task.split()
    mu, sd = feats[tr].mean(0), feats[tr].std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (feats - mu) / sd
    v = np.zeros(Z.shape[1]) if noise_var is None else noise_var / sd ** 2
    lam = task.N_train * np.maximum(v, sigma_floor ** 2)
    Xf = np.hstack([Z, np.ones((len(Z), 1))]); X, y = Xf[tr], task.y[tr]
    Lam = np.append(lam, task.N_train * sigma_floor ** 2)
    w = np.linalg.solve(X.T @ X + np.diag(Lam), X.T @ y)
    L = 0.5 * np.mean((y - X @ w) ** 2)
    yte = task.y[te]; pred = Xf[te] @ w
    return L, np.sqrt(np.mean((yte - pred) ** 2)) / np.std(yte)

def noise_var_vector(dev, task, cov, Nrep):
    """Training-window mean readout-noise variance of each raw feature for Nrep passes per setting."""
    tr, _ = task.split(); T = cov['vQ'].shape[0]; Nv = dev.Nv
    V = np.zeros((T, Nv, NOBS))
    for i, k in enumerate(('vQ', 'vP', 'vQ2', 'vP2', 'vsx', 'vsy')):
        V[:, :, i] = cov[k]
    return V.reshape(T, NOBS * Nv)[tr].mean(0) / Nrep

# ---------------------------------------------------------------- SNR-aware readout
SNR_THRESH = 0.25
def evaluate_snr(dev, task, feats, noise_var, thresh=SNR_THRESH):
    """Readout that uses the measured readout-noise variance of every feature (estimable from the spread over passes):
    features whose noise variance exceeds `thresh` times their training-window variance are dropped, and the remaining
    ones get the noise-matched ridge penalty of evaluate_nm. With noise_var = 0 it is the protocol readout."""
    tr, _ = task.split(); sd2 = feats[tr].var(0)
    keep = noise_var < thresh * np.where(sd2 > 1e-24, sd2, np.inf)
    if keep.sum() == 0: keep[:] = True
    return evaluate_nm(dev, task, feats[:, keep], noise_var[keep])

_ExpLoss_call = ExpLoss.__call__
def _call(self, theta):
    if getattr(self, 'readout', 'protocol') != 'snr' or self.Nrep is None:
        return _ExpLoss_call(self, theta)
    if self.budget is not None and self.n >= self.budget:
        raise StopIteration
    th = self.project(theta) if self.project is not None else np.asarray(theta, float)
    dv, thr = realize(self.dev, th, self.drift, self.rng)
    feats, extra, nmax = ResX(dv, thr, self.eta_inj, self.sig_phi).run_moments(self.task.f)
    Lclean = evaluate(dv, thr, self.task, feats=feats)[0]
    cov = single_run_cov(dv, feats, extra, self.eta_c, self.eta_e)
    Fn = add_readout_noise(dv, feats, cov, self.Nrep, self.rng); self.passes += 2 * self.Nrep
    L, nr = evaluate_snr(dv, self.task, Fn, noise_var_vector(dv, self.task, cov, self.Nrep))
    self.n += 1; self.trace.append((self.n, *th, L, Lclean, nmax, nr))
    return L
ExpLoss.__call__ = _call

# ---------------------------------------------------------------- squeezed input noise at the detector + LO angles
def bath_NM(theta, dev, eta_inj=1.0, sig_phi=0.0, nth=0.0):
    r, re, dth = theta
    N = eta_inj * np.sinh(re) ** 2 + nth
    M = eta_inj * np.cosh(re) * np.sinh(re) * dev.s * np.exp(1j * dth) * np.exp(-0.5 * sig_phi ** 2)
    return N, M

def input_variance(phi, N, M):
    """Quadrature variance of the (reflected) input field in X_phi = a e^{-i phi} + h.c.: 1 + 2N + 2 Re(<b b> e^{-2 i phi}),
    <b b> = -M for the dissipator of build_ext. (Revision 2 correction: the first revision used -M*, which gives the same
    variance for the Q and P settings but the mirror-image angle for rotated settings. -M follows from input-output theory for
    L = sqrt(kappa)(mu a + nu a^dag), b_in = mu b0 - nu b0^dag, and equals <a a> of the bath-only steady state; see
    check_input_phase().)"""
    return 1 + 2 * N - 2 * np.real(M * np.exp(-2j * phi))

def squeezed_angle(N, M):
    """LO angle of minimum input-noise variance."""
    return 0.5 * np.angle(M) if abs(M) > 0 else 0.0

def check_input_phase():
    """The bath-only steady state of D[L_a] is the squeezed vacuum of the input, so <a a>_ss must equal <b_in b_in> = -M."""
    from common import DEV
    dv = replace(DEV, Nc=14, g=0.0, omega=0.0, omega_q=0.0)
    for th in [(0.0, 0.4, -1.626), (0.0, 0.3, 0.9)]:
        L0 = build_ext(dv, th)[0]; w, v = np.linalg.eig(L0); D = 2 * (dv.Nc + 1)
        rho = v[:, np.argmin(abs(w))].reshape(D, D); rho /= np.trace(rho)      # row-major vec, as _sop
        a = _ops(dv.Nc)[0]; N, M = bath_NM(th, dv)
        assert abs(np.trace(a @ a @ rho) + M) < 1e-4, (np.trace(a @ a @ rho), -M)
    print('check_input_phase ok')

class ResLO(ResX):
    """ResX whose two cavity 'quadrature' features are X_{phi1} and X_{phi2} (default Q and P), with the input-noise
    variance of each LO angle stored for the readout-noise model."""
    def __init__(self, dev, theta, eta_inj=1.0, sig_phi=0.0, nth=0.0, lo=(0.0, np.pi / 2)):
        super().__init__(dev, theta, eta_inj, sig_phi, nth)
        r, re, dth = theta
        al, sig = _ops(dev.Nc); ch, sh = np.cosh(r), np.sinh(r)
        a = ch * al + dev.s * sh * al.conj().T; ad = a.conj().T
        X1 = a * np.exp(-1j * lo[0]) + ad * np.exp(1j * lo[0]); X2 = a * np.exp(-1j * lo[1]) + ad * np.exp(1j * lo[1])
        sx = sig + sig.conj().T; sy = 1j * (sig.conj().T - sig)
        ops = (X1, X2, X1 @ X1, X2 @ X2, sx, sy, X1 @ X1 @ X1, X1 @ X1 @ X1 @ X1, X2 @ X2 @ X2, X2 @ X2 @ X2 @ X2)
        obs = np.array([O.T.ravel() for O in ops])
        self.obs_all = np.asarray(obs @ self.B).real; self.obs = self.obs_all[:NOBS]
        N, M = bath_NM(theta, dev, eta_inj, sig_phi, nth); self.lo = lo
        self.Vin = (input_variance(lo[0], N, M), input_variance(lo[1], N, M))

def single_run_cov_sq(dev, feats, extra, Vin, eta_c=0.8, eta_e=0.5):
    """As single_run_cov, but the white detector noise of each cavity setting is set by the input-field variance at its LO
    angle: s = (eta_c V_in + 1 - eta_c) / (eta_c kappa tau). Vin = (1, 1) reproduces single_run_cov."""
    T = feats.shape[0]; Nv = dev.Nv
    F = feats.reshape(T, Nv, NOBS); X1, X2, X12, X22, sx, sy = [F[:, :, i] for i in range(NOBS)]
    A3, A4, B3, B4 = [extra[:, :, i] for i in range(4)]
    tau = dev.T_in / Nv
    s1 = (eta_c * Vin[0] + 1 - eta_c) / (eta_c * dev.kappa * tau); s2 = (eta_c * Vin[1] + 1 - eta_c) / (eta_c * dev.kappa * tau)
    se = 1.0 / (eta_e * dev.gamma * tau)
    return dict(vQ=X12 - X1 ** 2 + s1, vP=X22 - X2 ** 2 + s2,
                vQ2=np.maximum(A4 - X12 ** 2, 0) + 4 * X12 * s1 + 2 * s1 ** 2, vP2=np.maximum(B4 - X22 ** 2, 0) + 4 * X22 * s2 + 2 * s2 ** 2,
                cQ=A3 - X1 * X12 + 2 * X1 * s1, cP=B3 - X2 * X22 + 2 * X2 * s2, vsx=1 - sx ** 2 + se, vsy=1 - sy ** 2 + se, sc=(s1, s2), se=se)

def selftest_lo():
    from common import DEV, MG
    th = (0.1, 0.4, 0.5); f = MG.f[:300]
    A, EA, _ = ResX(DEV, th).run_moments(f); B, EB, _ = ResLO(DEV, th).run_moments(f)
    assert np.abs(A - B).max() < 1e-10 and np.abs(EA - EB).max() < 1e-10
    ca = single_run_cov(DEV, A, EA); cb = single_run_cov_sq(DEV, B, EB, (1.0, 1.0))
    assert all(np.allclose(ca[k], cb[k]) for k in ('vQ', 'vP', 'vQ2', 'vP2', 'cQ', 'cP', 'vsx'))
    N, M = bath_NM(th, DEV); ps = squeezed_angle(N, M)
    assert abs(input_variance(ps, N, M) - np.exp(-2 * th[1])) < 1e-9, (input_variance(ps, N, M), np.exp(-2 * th[1]))
    print('selftest_lo ok; V_min', input_variance(ps, N, M), 'V_max', input_variance(ps + np.pi / 2, N, M))
