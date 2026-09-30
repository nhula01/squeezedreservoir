"""Squeezed cavity-QED quantum reservoir computer.

Model (frame rotating at half the parametric pump frequency, phi_s = 0):
  H0 = w a^dag a + wq s^dag s - (lam/2)(a^2 + a^dag^2) + g(a^dag s + a s^dag),  tanh(2r) = lam/w
  Hin = i eps f(t) (a - a^dag)
  drho = -i[H, rho] + D[L_a] rho + gam D[s] rho,  L_a = sqrt(kappa)(cosh re a + e^{i dth} sinh re a^dag)
Simulated in the Bogoliubov basis a = cosh r alpha + sinh r alpha^dag, in which the quadratic cavity part
is Omega alpha^dag alpha with Omega = w / cosh 2r. Fock truncation is applied to alpha.
Controls: theta = (r, re, dth). Fabricated: g, kappa, gamma, eps, w, wq.
"""
import numpy as np
import scipy.linalg as sla
from dataclasses import dataclass, replace

NOBS = 6   # Q, P, Q^2, P^2, sigma_x, sigma_y

# ---------------------------------------------------------------- device
@dataclass
class Device:
    omega: float = 1.0      # cavity detuning from pump/2 (sets units)
    omega_q: float = 1.0    # emitter detuning from pump/2
    g: float = 0.5          # emitter-cavity coupling (fabricated)
    kappa: float = 0.2      # cavity decay (fabricated)
    gamma: float = 0.1      # emitter decay (fabricated)
    eps: float = 0.3        # drive amplitude per unit encoded input f in [-1, 1]
    T_in: float = 2.0       # duration of one input symbol
    Nv: int = 4             # virtual nodes (samples) per symbol
    Nc: int = 10            # Fock cutoff (levels 0..Nc) in the Bogoliubov mode
    K: int = 7              # Chebyshev nodes in the input
    sigma_rel: float = 0.01  # relative measurement precision assumed by the readout (sets the ridge)
    s: float = -1.0         # sign of the Bogoliubov mixing = parametric phase phi_s (s=+1: phi_s=0, s=-1: phi_s=pi)
    phi_d: float = 0.0      # drive phase relative to the parametric axis: eps_eff = eps*sqrt(cosh(2r) - s*sinh(2r)cos(2 phi_d))

def _ops(Nc):
    N = Nc + 1
    al = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    I_c = np.eye(N)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)      # |g><e| , basis (g, e)
    I2 = np.eye(2)
    return np.kron(al, I2), np.kron(I_c, sm)

def _sop(A, B):           # superoperator of rho -> A rho B  (row-major vec)
    return np.kron(A, B.T)

def _dissipator(L):
    LdL = L.conj().T @ L
    I = np.eye(L.shape[0])
    return _sop(L, L.conj().T) - 0.5 * _sop(LdL, I) - 0.5 * _sop(I, LdL)

def _comm(H):
    I = np.eye(H.shape[0])
    return -1j * (_sop(H, I) - _sop(I, H))

def build(dev, theta):
    """Return L0, L1 (drive per unit f), observable vectors, photon-number vector, dim."""
    r, re, dth = theta
    al, sig = _ops(dev.Nc)
    ald, sigd = al.conj().T, sig.conj().T
    ch, sh = np.cosh(r), np.sinh(r)
    a = ch * al + dev.s * sh * ald             # phi_s = 0 (s=+1) or pi (s=-1)
    ad = a.conj().T
    Om = dev.omega / np.cosh(2 * r)
    H0 = Om * (ald @ al) + dev.omega_q * (sigd @ sig) + dev.g * (ad @ sig + a @ sigd)
    X = 1j * dev.eps * (a * np.exp(-1j * dev.phi_d) - ad * np.exp(1j * dev.phi_d))   # drive operator per unit f (phase phi_d)
    c1 = np.cosh(re) * ch + np.exp(1j * dth) * np.sinh(re) * sh
    c2 = dev.s * (np.cosh(re) * sh + np.exp(1j * dth) * np.sinh(re) * ch)
    La = np.sqrt(dev.kappa) * (c1 * al + c2 * ald)
    L0 = _comm(H0) + _dissipator(La) + dev.gamma * _dissipator(sig)
    L1 = _comm(X)
    Q = a + ad
    P = 1j * (ad - a)
    sx = sig + sigd
    sy = 1j * (sigd - sig)
    obs = np.array([O.T.ravel() for O in (Q, P, Q @ Q, P @ P, sx, sy)])
    nvec = (ad @ a).T.ravel()
    return L0, L1, obs, nvec, a.shape[0]

# ---------------------------------------------------------------- Chebyshev propagator in the input
def _cheb_nodes(K):
    k = np.arange(K)
    x = np.cos((2 * k + 1) * np.pi / (2 * K))
    w = (-1.0) ** k * np.sin((2 * k + 1) * np.pi / (2 * K))
    return x, w

def _cheb_coeffs(u, x, w):
    d = u[:, None] - x[None, :]
    hit = np.abs(d) < 1e-14
    d[hit] = 1.0
    c = w[None, :] / d
    c[hit.any(1)] = hit[hit.any(1)].astype(float)
    return c / c.sum(1, keepdims=True)

def _herm_basis(D):
    """Sparse orthonormal basis of Hermitian DxD matrices as columns of a (D^2 x D^2) complex matrix (row-major vec)."""
    import scipy.sparse as sp
    rows, cols, vals = [], [], []
    k = 0
    s = 1 / np.sqrt(2)
    for i in range(D):
        rows.append(i * D + i); cols.append(k); vals.append(1.0); k += 1
    for i in range(D):
        for j in range(i + 1, D):
            rows += [i * D + j, j * D + i]; cols += [k, k]; vals += [s, s]; k += 1
            rows += [i * D + j, j * D + i]; cols += [k, k]; vals += [1j * s, -1j * s]; k += 1
    return sp.csc_matrix((vals, (rows, cols)), shape=(D * D, D * D))

def _to_real(L, B):
    """Real matrix of the Hermiticity-preserving superoperator L in the basis B."""
    return np.asarray((B.conj().T @ (B.T.conj().T @ L.T).T)).real if False else np.asarray(B.conj().T @ (L @ B)).real

class Reservoir:
    """Reservoir at control point theta. Propagation is done in the real Hermitian basis."""
    def __init__(self, dev, theta):
        self.dev, self.theta = dev, np.asarray(theta, float)
        L0, L1, obs, nvec, D = build(dev, theta)
        self.D = D
        B = _herm_basis(D)
        L0r = np.asarray(B.conj().T @ (L0 @ B)).real
        L1r = np.asarray(B.conj().T @ (L1 @ B)).real
        self.obs = np.asarray(obs @ B).real          # Tr(O rho) = obs . vec(rho) = (obs B) . c
        self.nvec = np.asarray(nvec @ B).real
        self.B = B
        tau = dev.T_in / dev.Nv
        self.x, self.w = _cheb_nodes(dev.K)
        self.Pstack = np.stack([sla.expm((L0r + xk * L1r) * tau) for xk in self.x])
        self.Pflat = np.ascontiguousarray(self.Pstack.reshape(dev.K, -1))
        self.L0r, self.L1r, self.tau = L0r, L1r, tau
        self.rho0 = np.zeros(D * D); self.rho0[0] = 1.0   # alpha-vacuum, emitter ground (basis index 0 = E_00)

    def run(self, f, return_states=False):
        """f: encoded input in [-1,1], shape (T,). Returns features (T, 4*Nv), max photon number."""
        dev, D = self.dev, self.D
        T = len(f)
        rho = self.rho0.copy()
        C = _cheb_coeffs(np.asarray(f, float), self.x, self.w)
        feats = np.empty((T, NOBS * dev.Nv))
        nmax = 0.0
        states = [] if return_states else None
        obs, nvec, Pflat, D2 = self.obs, self.nvec, self.Pflat, D * D
        for t in range(T):
            P = (C[t] @ Pflat).reshape(D2, D2)
            for v in range(dev.Nv):
                rho = P @ rho
                feats[t, NOBS * v:NOBS * (v + 1)] = obs @ rho
                n = nvec @ rho
                if n > nmax:
                    nmax = n
            if return_states:
                states.append(rho.copy())
        if return_states:
            return feats, nmax, states
        return feats, nmax

    def to_matrix(self, c):
        return np.asarray(self.B @ c).reshape(self.D, self.D)

    def exact_step(self, rho, u):
        return sla.expm((self.L0r + u * self.L1r) * self.tau) @ rho

# ---------------------------------------------------------------- tasks
def mackey_glass(N, beta=0.2, gamma=0.1, n=10, tau=17, transient=1000, x0=1.2, seed=None):
    """Unit-step Euler integration of the MG delay equation, transient discarded."""
    tot = N + transient
    x = np.zeros(tot + tau + 1)
    x[:tau + 1] = x0
    for t in range(tau, tot + tau):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau] ** n) - gamma * x[t]
    return x[tau + 1 + transient:]

def narma10(N, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, N + 20)
    y = np.zeros(N + 20)
    for t in range(9, N + 19):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1]) + 1.5 * u[t - 9] * u[t] + 0.1
    return u[20:], y[20:]

@dataclass
class Task:
    name: str
    u: np.ndarray            # raw input series
    y: np.ndarray            # target aligned with u (same index)
    N_fade: int
    N_train: int
    horizon: int = 0

    @property
    def f(self):             # affine encoding of the input to [-1, 1]
        u = self.u
        return 2 * (u - u.min()) / (u.max() - u.min()) - 1

    def split(self):
        T = len(self.u) - self.horizon
        tr = slice(self.N_fade, self.N_fade + self.N_train)
        te = slice(self.N_fade + self.N_train, T)
        return tr, te

def make_mg(N_total=1000, N_fade=100, N_train=700, horizon=10):
    x = mackey_glass(N_total)
    y = np.roll(x, -horizon)
    return Task("mg", x, y, N_fade, N_train, horizon)

def make_narma(N_total=1000, N_fade=10, N_train=590, seed=0):
    u, y = narma10(N_total, seed)
    return Task("narma10", u, y, N_fade, N_train, 0)

# ---------------------------------------------------------------- readout and loss
def ridge_fit(X, y, delta=1e-10):
    """Minimizes 1/2 ||y - Xw||^2 + delta/2 ||w||^2 via SVD (bias column included in X)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    coef = s / (s ** 2 + delta)
    return Vt.T @ (coef * (U.T @ y))

def evaluate(dev, theta, task, delta=None, feats=None, noise=None, rng=None, return_all=False):
    """Task loss L (train), held-out NRMSE, photon max, w*, X.
    Features are standardized on the training window; the ridge delta defaults to N_train*sigma_rel^2 with
    sigma_rel = dev.sigma_rel (measurement precision relative to the feature scale).
    noise: optional relative additive Gaussian noise (fraction of the training std of each feature)."""
    nmax = np.nan
    if feats is None:
        res = Reservoir(dev, theta)
        feats, nmax = res.run(task.f)
    tr, te = task.split()
    mu, sd = feats[tr].mean(0), feats[tr].std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (feats - mu) / sd
    if noise is not None:
        rng = rng or np.random.default_rng()
        Z = Z + noise * rng.standard_normal(Z.shape)
    if delta is None:
        delta = task.N_train * dev.sigma_rel ** 2
    Xf = np.hstack([Z, np.ones((len(Z), 1))])
    X, y = Xf[tr], task.y[tr]
    w = ridge_fit(X, y, delta)
    L = 0.5 * np.mean((y - X @ w) ** 2)
    yte = task.y[te]
    pred = Xf[te] @ w
    nrmse = np.sqrt(np.mean((yte - pred) ** 2)) / np.std(yte)
    if return_all:
        return L, nrmse, nmax, w, Z
    return L, nrmse, nmax

# ---------------------------------------------------------------- control box
BOX = np.array([[0.0, 0.5], [0.0, 0.5], [-np.pi, np.pi]])

def project(theta, box=BOX):
    th = np.array(theta, float)
    th[0] = np.clip(th[0], *box[0])
    th[1] = np.clip(th[1], *box[1])
    th[2] = (th[2] + np.pi) % (2 * np.pi) - np.pi   # phase is periodic
    return th

# ---------------------------------------------------------------- additional tasks
def lorenz63(N, dt=0.05, sub=5, transient=2000, seed=0):
    """x-component of Lorenz-63 (sigma=10, rho=28, beta=8/3), RK4 with step dt/sub, sampled every dt."""
    s, r_, b = 10.0, 28.0, 8.0 / 3.0
    def f(v):
        x, y, z = v
        return np.array([s * (y - x), x * (r_ - z) - y, x * y - b * z])
    v = np.array([1.0, 1.0, 1.0]); h = dt / sub; out = np.zeros(N + transient)
    for n in range(N + transient):
        for _ in range(sub):
            k1 = f(v); k2 = f(v + h / 2 * k1); k3 = f(v + h / 2 * k2); k4 = f(v + h * k3)
            v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[n] = v[0]
    return out[transient:]

def make_lorenz(N_total=1000, N_fade=100, N_train=700, horizon=10):
    x = lorenz63(N_total)
    y = np.roll(x, -horizon)
    return Task("lorenz", x, y, N_fade, N_train, horizon)

def nce(N, seed=0, snr_db=24):
    """Nonlinear channel equalization (Jaeger & Haas 2004): input u(n), target d(n-2)."""
    rng = np.random.default_rng(seed)
    d = rng.choice([-3.0, -1.0, 1.0, 3.0], N + 20)
    q = np.zeros(N + 20)
    for n in range(7, N + 18):
        q[n] = (0.08 * d[n + 2] - 0.12 * d[n + 1] + d[n] + 0.18 * d[n - 1] - 0.1 * d[n - 2] + 0.091 * d[n - 3]
                - 0.05 * d[n - 4] + 0.04 * d[n - 5] + 0.03 * d[n - 6] + 0.01 * d[n - 7])
    u = q + 0.036 * q ** 2 - 0.011 * q ** 3
    u = u + rng.standard_normal(len(u)) * np.std(u) * 10 ** (-snr_db / 20)
    y = np.roll(d, 2)                                   # target d(n-2) aligned with u(n)
    return u[10:N + 10], y[10:N + 10]

def make_nce(N_total=1000, N_fade=10, N_train=690, seed=0):
    u, y = nce(N_total, seed)
    return Task("nce", u, y, N_fade, N_train, 0)

# ---------------------------------------------------------------- effective device parameters
def effective_parameters(dev, theta, T=80.0, dt=0.25):
    """Effective parameters of the reservoir at control theta (no input drive):
    eps_eff = eps e^{-r}; Omega = w/cosh 2r; g_co = g cosh r; g_cr = g sinh r;
    emitter relaxation rates gamma_x, gamma_y, gamma_z from the integrated relaxation time of <sigma_i>
    toward its stationary value; stationary Var(Q), Var(P), <a^dag a>, <sigma^dag sigma>."""
    r, re, dth = theta
    res = Reservoir(dev, theta)
    L0 = res.L0r; D = res.D
    P = sla.expm(L0 * dt); n = int(T / dt)
    al, sig = _ops(dev.Nc); ald, sigd = al.conj().T, sig.conj().T
    sx, sy = sig + sigd, 1j * (sigd - sig); sz = sigd @ sig - sig @ sigd
    ch, sh = np.cosh(r), np.sinh(r); a = ch * al + dev.s * sh * ald; ad = a.conj().T
    Q, Pq = a + ad, 1j * (ad - a)
    B = res.B
    vec = lambda O: np.asarray(O.T.ravel() @ B).real
    obs = {k: vec(O) for k, O in dict(sx=sx, sy=sy, sz=sz, Q=Q, P=Pq, Q2=Q @ Q, P2=Pq @ Pq, n=ad @ a, ne=sigd @ sig).items()}
    # stationary state: propagate the vacuum-ground state for a long time
    rho = res.rho0.copy()
    Plong = sla.expm(L0 * 400.0); rho_ss = Plong @ rho
    ss = {k: float(v @ rho_ss) for k, v in obs.items()}
    out = dict(eps_eff=dev.eps * np.sqrt(np.cosh(2 * r) - dev.s * np.sinh(2 * r) * np.cos(2 * dev.phi_d)), Omega=dev.omega / np.cosh(2 * r), g_co=dev.g * ch, g_cr=dev.g * sh,
               VarQ=ss['Q2'] - ss['Q'] ** 2, VarP=ss['P2'] - ss['P'] ** 2, n_ss=ss['n'], ne_ss=ss['ne'])
    # relaxation rates: emitter prepared in +x, +y, excited on top of the stationary cavity state
    rho_ss_m = res.to_matrix(rho_ss)
    I2 = np.eye(2); N = dev.Nc + 1
    def prep(op):   # replace the emitter part of the stationary state by |psi><psi|
        rc = np.einsum('iajb->ij', rho_ss_m.reshape(N, 2, N, 2))  # cavity marginal
        return np.kron(rc, op)
    for key, psi in (('x', np.array([1, 1]) / np.sqrt(2)), ('y', np.array([1, 1j]) / np.sqrt(2)), ('z', np.array([0, 1.0]))):
        rho0 = prep(np.outer(psi, psi.conj()))
        c = np.asarray(B.conj().T @ rho0.ravel()).real.ravel()
        o = obs['s' + key]; s_inf = ss['s' + key]; s0 = float(o @ c)
        integ = 0.0; cur = c
        for _ in range(n):
            integ += abs(float(o @ cur) - s_inf) * dt; cur = P @ cur
        out['gamma_' + key] = abs(s0 - s_inf) / integ if integ > 0 else np.nan
    return out
