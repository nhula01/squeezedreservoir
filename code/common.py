import numpy as np, time, json, os, sys
from sqz import *
from scipy.optimize import minimize

DEV = Device(omega=1.0, omega_q=1.0, g=0.5, kappa=0.2, gamma=0.1, eps=0.2, T_in=2.0, Nv=4, Nc=8, K=7, s=1.0, sigma_rel=0.01)
MG = make_mg(N_total=1000, N_fade=100, N_train=700, horizon=10)
NARMA = make_narma(N_total=1000, N_fade=10, N_train=590, seed=0)
H_FD = np.array([1e-2, 1e-2, 1e-2])          # default finite-difference steps (r, re, dtheta[rad])
SCALE = np.array([0.5, 0.5, np.pi])          # box half-widths used to normalize optimizer coordinates
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA, exist_ok=True)

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

class Counter:
    """Wraps evaluate(); counts every reservoir evaluation and records the trace."""
    def __init__(self, dev, task, noise=None, rng=None, budget=None):
        self.dev, self.task, self.noise, self.rng, self.budget = dev, task, noise, rng, budget
        self.n = 0
        self.trace = []                      # (n, theta, L, nrmse, nmax)

    def __call__(self, theta):
        if self.budget is not None and self.n >= self.budget:
            raise StopIteration
        th = project(theta)
        L, nr, nm = evaluate(self.dev, th, self.task, noise=self.noise, rng=self.rng)
        self.n += 1
        self.trace.append((self.n, *th, L, nr, nm))
        return L

    def best_so_far(self, budget):
        tr = np.array(self.trace)
        Ls = tr[:, 4]
        out = np.full(budget, np.nan)
        best = np.inf
        for i in range(min(budget, len(Ls))):
            best = min(best, Ls[i]); out[i] = best
        out[len(Ls):] = best
        return out

def fd_gradient(fun, theta, h=H_FD, active=(0, 1, 2)):
    g = np.zeros(3)
    for i in active:
        e = np.zeros(3); e[i] = h[i]
        g[i] = (fun(theta + e) - fun(theta - e)) / (2 * h[i])
    return g

def gradient_descent(fun, theta0, budget, h=H_FD, s0=0.15, max_bt=4, armijo=1e-4, record=None, active=(0, 1, 2)):
    """Projected gradient descent with finite-difference gradients and backtracking, in box-normalized coordinates.
    Every loss evaluation goes through fun (which counts). Returns list of accepted (theta, L)."""
    theta = project(theta0)
    L = fun(theta)
    path = [(theta.copy(), L, 0)]
    eta = None
    while True:
        try:
            g = fd_gradient(fun, theta, h, active)
        except StopIteration:
            break
        gz = g * SCALE                          # gradient wrt normalized coords z = theta/SCALE
        gn = np.linalg.norm(gz)
        if gn == 0:
            break
        if eta is None:
            eta = s0 / gn
        else:
            eta *= 2.0
        accepted = False
        try:
            for _ in range(max_bt):
                cand = project(theta - eta * gz * SCALE)
                Lc = fun(cand)
                if Lc < L - armijo * eta * gn ** 2:
                    theta, L, accepted = cand, Lc, True
                    break
                eta *= 0.5
        except StopIteration:
            break
        if record is not None:
            record(theta, L)
        path.append((theta.copy(), L, 0))
        if not accepted:
            eta *= 0.5
            if eta * gn < 1e-5:
                break
    return path

def fd_gradient_conv(fun, theta, h0=H_FD, tol=0.05, max_halvings=3, box=BOX, active=(0, 1, 2), return_info=False):
    """Step-adaptive, Richardson-extrapolated finite-difference gradient with an error estimate.
    Interior directions: central differences at h and h/2, D = (4 D(h/2) - D(h))/3, error = |D(h/2) - D(h)|/3.
    Directions within h of a box face (r, re): second-order one-sided differences inward at h and h/2 (no projection).
    The step is halved while the error estimate exceeds tol*|D| (up to max_halvings). Every evaluation goes through fun."""
    theta = np.asarray(theta, float); g = np.zeros(3); err = np.zeros(3); hs = np.zeros(3); cache = {}
    def f(th):
        key = tuple(np.round(th, 12))
        if key not in cache: cache[key] = fun(th)
        return cache[key]
    f0 = f(theta)
    for i in active:
        h = h0[i]
        for _ in range(max_halvings + 1):
            e = np.zeros(3); e[i] = 1.0
            lo, hi = (box[i] if i < 2 else (-np.inf, np.inf))
            def D(hh):
                if theta[i] - hh >= lo and theta[i] + hh <= hi:
                    return (f(theta + hh * e) - f(theta - hh * e)) / (2 * hh)
                sgn = 1.0 if theta[i] - hh < lo else -1.0            # one-sided, second order, inward
                return sgn * (-3 * f0 + 4 * f(theta + sgn * hh * e) - f(theta + 2 * sgn * hh * e)) / (2 * hh)
            d1, d2 = D(h), D(h / 2)
            g[i] = (4 * d2 - d1) / 3; err[i] = abs(d2 - d1) / 3; hs[i] = h
            if err[i] <= tol * abs(g[i]) or abs(g[i]) < 1e-12: break
            h /= 2
    return (g, err, hs) if return_info else g

def gradient_descent_conv(fun, theta0, budget, s0=0.15, max_bt=4, armijo=1e-4, record=None, active=(0, 1, 2), tol=0.05):
    """Projected gradient descent using fd_gradient_conv (Richardson, boundary-aware)."""
    theta = project(theta0); L = fun(theta); path = [(theta.copy(), L, 0)]; eta = None
    while True:
        try:
            g = fd_gradient_conv(fun, theta, tol=tol, active=active)
        except StopIteration:
            break
        gz = g * SCALE; gn = np.linalg.norm(gz)
        if gn == 0: break
        eta = s0 / gn if eta is None else eta * 2.0
        accepted = False
        try:
            for _ in range(max_bt):
                cand = project(theta - eta * gz * SCALE); Lc = fun(cand)
                if Lc < L - armijo * eta * gn ** 2:
                    theta, L, accepted = cand, Lc, True; break
                eta *= 0.5
        except StopIteration:
            break
        if record is not None: record(theta, L)
        path.append((theta.copy(), L, 0))
        if not accepted:
            eta *= 0.5
            if eta * gn < 1e-5: break
    return path

def random_search(fun, budget, rng):
    while True:
        th = np.array([rng.uniform(0, 0.5), rng.uniform(0, 0.5), rng.uniform(-np.pi, np.pi)])
        try:
            fun(th)
        except StopIteration:
            break

def nelder_mead(fun, theta0, budget):
    z0 = np.asarray(theta0) / SCALE
    def fz(z):
        return fun(z * SCALE)
    try:
        minimize(fz, z0, method='Nelder-Mead',
                 bounds=[(0, 1), (0, 1), (-1, 1)],
                 options=dict(maxfev=budget, initial_simplex=np.array([z0, z0 + [0.15, 0, 0], z0 + [0, 0.15, 0], z0 + [0, 0, 0.15]]),
                              xatol=1e-8, fatol=1e-12))
    except StopIteration:
        pass

def random_init(rng, n):
    return np.column_stack([rng.uniform(0, 0.5, n), rng.uniform(0, 0.5, n), rng.uniform(-np.pi, np.pi, n)])
