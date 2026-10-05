"""Revision 2: is phase-sensitive physics of any use at the best re-fabricated device?

Squeezing changes the reservoir in two ways: it renormalizes ordinary parameters (cavity frequency, co-rotating coupling,
input strength), which re-fabrication can reproduce, and it adds phase-sensitive terms that no ordinary Jaynes-Cummings
device has. Here the phase-sensitive terms are given to the ordinary device directly, as independent knobs in the lab frame
(drive phase fixed at 0, which fixes the phase reference):
    H_ps = -(lam/2)(e^{-i phi_s} a^2 + h.c.)            degenerate parametric (intracavity squeezing) term, lam < omega_a
         + g_cr (e^{i phi_c} a sigma + h.c.)               counter-rotating (two-mode-squeezing) cavity-emitter coupling
    bath  D[L_a], L_a = sqrt(kappa)(cosh r_e a + e^{i dth} sinh r_e a^dag)   squeezed bath (r_e, dth)
The squeezed device of the paper is the special case g_cr = 0 (its counter-rotating coupling appears only in the Bogoliubov
frame), so these knobs contain everything squeezing can do and more: g_cr is free, and lam, r_e can exceed the paper's box.
Ranges: lam/omega_a in [0, 0.9], g_cr in [0, 1], r_e in [0, 0.8], all phases free; photon budget 2.5, N_c = 10, checks at 14;
measurement-aware objective of rev_bounds2.py (readout model at 1e7 passes, noise-matched ridge, input-noise detector model).

Studies
  opt   at the best re-fabricated device of each task (wide box): (a) first-order response, the loss change for a small
        amplitude of each phase-sensitive term at 8 phases; (b) search over the six knobs with the device fixed (scans of each
        mechanism, then Nelder-Mead, 150 evaluations); (c) joint local search over the six ordinary parameters and the six
        knobs from the best point of (b) (Nelder-Mead, 150 evaluations).
  path  on the straight line (in the normalized coordinates of the wide box) from the tuned base device of Sec. 2.4 to the
        best re-fabricated device, at 1/3 and 2/3: ordinary loss and the knob search (b). Gain against the distance from the
        re-fabricated optimum.
Usage: python3 rev_ps.py opt nce narma ... | python3 rev_ps.py path nce ...   -> data/rev_ps_<task>.npz
       python3 rev_ps.py check     (the generalized model reproduces the squeezed device of the paper)"""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *
from sqz import _ops, _comm
from scipy.optimize import minimize
from rev_hard import LADDER
NC = 10
KN = ['lam', 'phi_s', 'g_cr', 'phi_c', 'r_e', 'dth']
KLO = np.array([0.0, -np.pi, 0.0, -np.pi, 0.0, -np.pi]); KHI = np.array([0.9, np.pi, 1.0, np.pi, 0.8, np.pi])   # lam as fraction of omega_a

class ResPS(ResLO):
    """ResLO (squeezed bath, readout moments, input-noise variance) plus the phase-sensitive Hamiltonian H_ps, r = 0."""
    def __init__(self, dev, ps):
        lamf, phis, gcr, phic, re, dth = ps
        super().__init__(dev, (0.0, re, dth))
        al, sig = _ops(dev.Nc); ald, sigd = al.conj().T, sig.conj().T
        lam = lamf * abs(dev.omega)
        H = -0.5 * lam * (np.exp(-1j * phis) * al @ al + np.exp(1j * phis) * ald @ ald) + gcr * (np.exp(1j * phic) * al @ sig + np.exp(-1j * phic) * ald @ sigd)
        if lam != 0 or gcr != 0:
            Lh = np.asarray(self.B.conj().T @ (_comm(H) @ self.B)).real
            self.L0r = self.L0r + Lh
            self.Pflat = np.ascontiguousarray(np.stack([sla.expm((self.L0r + xk * self.L1r) * self.tau) for xk in self.x]).reshape(dev.K, -1))

import scipy.linalg as sla

def evaluate_ps(dev, ps, task):
    res = ResPS(dev, ps); F, E, nm = res.run_moments(task.f)
    cov = single_run_cov_sq(dev, F, E, res.Vin); nv = noise_var_vector(dev, task, cov, r2.NREP_OBJ)
    L, nr = evaluate_nm(dev, task, F, nv)
    return L, nr, nm

def kp(v): return KLO + (KHI - KLO) * np.clip(v, 0, 1)
def kv(p): return (np.asarray(p, float) - KLO) / (KHI - KLO)

class KObj:
    """Counted objective over the knobs (device fixed) or over device + knobs (joint). Rows: 6 device params, 6 knobs, L, NRMSE, nmax."""
    def __init__(self, dev, task, budget, joint=False):
        self.dev, self.task, self.budget, self.joint, self.n, self.trace = dev, task, budget, joint, 0, []
    def __call__(self, v):
        if self.n >= self.budget: raise StopIteration
        if self.joint:
            p = rb.to_p(v[:6]); dv = replace(self.dev, **dict(zip(rb.PN, p))); k = kp(v[6:])
        else:
            dv = self.dev; p = np.array([dv.g, dv.kappa, dv.gamma, dv.omega, dv.omega_q, dv.eps]); k = kp(v)
        L, nr, nm = evaluate_ps(dv, k, self.task); self.n += 1; self.trace.append([*p, *k, L, nr, nm])
        return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))

def best(tr):
    tr = np.asarray(tr); ok = tr[:, 14] <= NMAX; return tr[np.where(ok)[0][tr[ok, 12].argmin()]]

def first_order(dev, task, L0, amp=(0.05, 0.05, 0.05), nph=8):
    """Loss change for a small amplitude of each mechanism (lam/omega, g_cr, r_e) at nph phases; rows [mech, phase, dL/L]."""
    rows = []
    for m, (ia, ip) in enumerate(((0, 1), (2, 3), (4, 5))):
        for ph in np.linspace(-np.pi, np.pi, nph, endpoint=False):
            k = np.zeros(6); k[ia] = amp[m]; k[ip] = ph
            rows.append([m, ph, evaluate_ps(dev, k, task)[0] / L0 - 1])
    return np.array(rows)

def knob_search(dev, task, budget=150):
    """Scans of each mechanism alone (3 amplitudes x 4 phases each), then Nelder-Mead over all six knobs from the best."""
    obj = KObj(dev, task, budget); obj(kv(np.zeros(6)))
    for ia, ip, amps in ((0, 1, (0.2, 0.5, 0.8)), (2, 3, (0.1, 0.3, 0.7)), (4, 5, (0.15, 0.35, 0.6))):
        for a_ in amps:
            for ph in (-np.pi / 2, 0.0, np.pi / 2, np.pi - 1e-9):
                k = np.zeros(6); k[ia] = a_; k[ip] = ph; obj(kv(k))
    T = np.array(obj.trace); pen = np.where(T[:, 14] <= NMAX, T[:, 12], np.inf); v0 = kv(T[pen.argmin(), 6:12])
    simplex = np.vstack([v0] + [np.clip(v0 + 0.1 * np.eye(6)[k] * (1 if v0[k] < 0.5 else -1), 0, 1) for k in range(6)])
    try: minimize(obj, v0, method='Nelder-Mead', bounds=[(0, 1)] * 6, options=dict(maxfev=10 ** 6, initial_simplex=simplex, xatol=1e-5, fatol=1e-12))
    except StopIteration: pass
    return np.array(obj.trace)

def joint_local(dev, task, k0, budget=150):
    obj = KObj(dev, task, budget, joint=True)
    u0 = np.concatenate([rb.to_u([dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps]), kv(k0)])
    simplex = np.vstack([u0] + [np.clip(u0 + 0.04 * np.eye(12)[k] * (1 if u0[k] < 0.5 else -1), 0, 1) for k in range(12)])
    try: minimize(obj, u0, method='Nelder-Mead', bounds=[(0, 1)] * 12, options=dict(maxfev=10 ** 6, initial_simplex=simplex, xatol=1e-5, fatol=1e-12))
    except StopIteration: pass
    return np.array(obj.trace)

def masked_local(dev, task, u_base, free, budget=150, step=0.05):
    """Nelder-Mead over the coordinates `free` of the 12-vector (6 device coordinates of the wide box, 6 knob coordinates),
    the others held at u_base. Returns the trace (rows as KObj)."""
    obj = KObj(dev, task, budget, joint=True); free = np.array(free)
    def f(x):
        u = u_base.copy(); u[free] = x; return obj(u)
    x0 = u_base[free]
    simplex = np.vstack([x0] + [np.clip(x0 + step * np.eye(len(free))[k] * (1 if x0[k] < 0.5 else -1), 0, 1) for k in range(len(free))])
    try: minimize(f, x0, method='Nelder-Mead', bounds=[(0, 1)] * len(free), options=dict(maxfev=10 ** 6, initial_simplex=simplex, xatol=1e-5, fatol=1e-12))
    except StopIteration: pass
    return np.array(obj.trace)

def first_rows(F, L0, base_row):
    """First-order evaluations as trace rows (knob vectors reconstructed; amplitude 0.05)."""
    rows = []
    for m, ph, dl in F:
        k = np.zeros(6); ia, ip = ((0, 1), (2, 3), (4, 5))[int(m)]; k[ia] = 0.05; k[ip] = ph
        r = base_row.copy(); r[6:12] = k; r[12] = L0 * (1 + dl); r[14] = 0.0; rows.append(r)
    return np.array(rows)

def decomp(dev, task, knob_trace, budget=150):
    """Which phase-sensitive ingredient carries the joint gain? Three local searches from the re-fabricated device:
    'ord' six ordinary parameters only (knobs off; control for an unconverged ordinary optimum); 'sq' ordinary + the
    squeezing-accessible knobs (lam, phi_s, r_e, dth; g_cr = 0); 'cr' ordinary + the counter-rotating coupling only
    (g_cr, phi_c; lam = r_e = 0). Knob searches start from the best scan point of their subset."""
    ud = rb.to_u([dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps]); T = np.asarray(knob_trace)
    pen = np.where(T[:, 14] <= NMAX, T[:, 12], np.inf); out = {}
    for tag, free_k, sel in (('ord', [], lambda r: np.all(r[6:12] == 0)), ('sq', [0, 1, 4, 5], lambda r: r[8] == 0), ('cr', [2, 3], lambda r: r[6] == 0 and r[10] == 0)):
        idx = [i for i in range(len(T)) if sel(T[i])]; i0 = idx[int(np.argmin(pen[idx]))]
        u = np.concatenate([ud, kv(T[i0, 6:12])]); free = list(range(6)) + [6 + k for k in free_k]
        out[tag] = masked_local(dev, task, u, free, budget)
    return out

def check(dev, k, task, nm):
    out = [evaluate_ps(replace(dev, Nc=14), k, task)]
    if nm > 0.8 * NMAX: out.append(evaluate_ps(replace(dev, Nc=18), k, task))
    return np.array(out)

def refab_device(name):
    """Best re-fabricated (wide box, measurement-aware) ordinary device; for mg20 the best of the three searches."""
    if name in LADDER:
        c = [rb.best_row(np.load(os.path.join(DATA, f'rev_hard_{name}.npz'), allow_pickle=True)['ord_trace'])]
        for ext in ('conv', 'b'):
            f = os.path.join(DATA, f'rev_hard_{name}_{ext}.npz')
            if os.path.exists(f): c.append(np.load(f, allow_pickle=True)['ord_best'])
        p = min(c, key=lambda r: r[10])[:6]; task = LADDER[name][1]()
    else:
        p = rb.best_row(np.load(os.path.join(DATA, f'rev_bounds2_{name}.npz'), allow_pickle=True)['ord_trace'])[:6]; task = TASKS[name]
    return replace(DEV, Nc=NC, phi_d=0.0, **dict(zip(rb.PN, p))), task

if __name__ == '__main__':
    import sys
    mode = sys.argv[1]
    if mode == 'check':
        task = TASKS['narma']; dv = replace(base_device2('narma', 14), phi_d=0.0)
        for r_, re_, dth_ in ((0.1, 0.2, 0.7), (0.15, 0.0, 0.0), (0.0, 0.3, -1.0)):
            a = r2.evaluate_meas(dv, (r_, re_, dth_), task); b = evaluate_ps(dv, (np.tanh(2 * r_), 0.0, 0.0, 0.0, re_, dth_), task)
            log('squeezed device (r, re, dth) = (%.2f, %.2f, %.2f): L %.6e, generalized model %.6e (rel. diff %.1e)' % (r_, re_, dth_, a[0], b[0], abs(a[0] / b[0] - 1)))
        sys.exit()
    for name in sys.argv[2:]:
        fp = os.path.join(DATA, f'rev_ps_{name}.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
        save = lambda: np.savez(fp, **out)
        dev, task = refab_device(name)
        if mode == 'opt':
            if 'L0' not in out:
                out['L0'] = np.array(evaluate_ps(dev, np.zeros(6), task)); save()
            L0 = out['L0'][0]
            if 'first' not in out:
                out['first'] = first_order(dev, task, L0); save()
            F = out['first']
            log(name, 'first order (small amplitude 0.05): best dL/L per mechanism (lam, g_cr, r_e):', [round(float(F[F[:, 0] == m, 2].min()), 5) for m in range(3)])
            if 'knob_trace' not in out:
                out['knob_trace'] = knob_search(dev, task); save()
            bk = best(out['knob_trace'])
            log(name, 'knobs on the re-fabricated device:', dict(zip(KN, bk[6:12].round(3))), 'L %.4e -> %.4e (%.2f%%)' % (L0, bk[12], 100 * (1 - bk[12] / L0)))
            if 'joint_trace' not in out:
                out['joint_trace'] = joint_local(dev, task, bk[6:12]); save()
            bj = best(np.vstack([out['knob_trace'], out['joint_trace']]))
            if 'chk' not in out:
                out['chk0'] = check(dev, np.zeros(6), task, out['L0'][2]); out['chk_knob'] = check(dev, bk[6:12], task, bk[14])
                dj = replace(dev, **dict(zip(rb.PN, bj[:6]))); out['chk_joint'] = check(dj, bj[6:12], task, bj[14])
                # the joint point with the knobs switched off: what the re-optimized ordinary parameters alone give
                out['joint_off'] = np.array(evaluate_ps(dj, np.zeros(6), task)); save()
            c = lambda k: out[k][-1][0]
            log(name, 'checked: device %.4e, + knobs %.4e (%.2f%%), joint local %.4e (%.2f%%; same device, knobs off %.4e)' % (
                c('chk0'), c('chk_knob'), 100 * (1 - c('chk_knob') / c('chk0')), c('chk_joint'), 100 * (1 - c('chk_joint') / c('chk0')), out['joint_off'][0]))
        elif mode == 'sqfix':
            # squeezing on the re-fabricated device itself (device fixed), with the drive phase continuous: squeezing-type
            # knobs only (lam, phi_s, r_e, dth; g_cr = 0), Nelder-Mead from the best squeezing-type scan or first-order point
            if 'L0' not in out:
                out['L0'] = np.array(evaluate_ps(dev, np.zeros(6), task)); save()
            if 'first' not in out:
                out['first'] = first_order(dev, task, out['L0'][0]); save()
            if 'sqscan' not in out:                      # squeezing-type scans (as knob_search, without the g_cr block)
                obj = KObj(dev, task, 10 ** 6)
                for ia, ip, amps in ((0, 1, (0.2, 0.5, 0.8)), (4, 5, (0.15, 0.35, 0.6))):
                    for a_ in amps:
                        for ph in (-np.pi / 2, 0.0, np.pi / 2, np.pi - 1e-9):
                            k = np.zeros(6); k[ia] = a_; k[ip] = ph; obj(kv(k))
                out['sqscan'] = np.array(obj.trace); save()
            base_row = np.asarray(out['sqscan'])[0].copy(); base_row[6:12] = 0; base_row[12:15] = out['L0']
            KT = np.vstack([base_row[None, :], out['sqscan'], first_rows(out['first'], out['L0'][0], base_row)])
            KT = KT[KT[:, 8] == 0]
            if 'sqfix' not in out:
                ud = rb.to_u([dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps])
                out['sqfix'] = masked_local(dev, task, np.concatenate([ud, kv(KT[int(np.argmin(KT[:, 12])), 6:12])]), [6, 7, 10, 11], 100, step=0.08); save()
            if 'sqjoint' not in out and name in LADDER:   # squeezing-type knobs + device, for the difficulty ladders
                ud = rb.to_u([dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps]); T = np.vstack([KT, out['sqfix']])
                out['sqjoint'] = masked_local(dev, task, np.concatenate([ud, kv(T[int(np.argmin(np.where(T[:, 14] <= NMAX, T[:, 12], np.inf))), 6:12])]),
                                              [0, 1, 2, 3, 4, 5, 6, 7, 10, 11], 150); save()
            if 'chk0' not in out:
                out['chk0'] = check(dev, np.zeros(6), task, out['L0'][2]); save()
            b = best(np.vstack([KT, out['sqfix']])); out['sqfix_chk'] = check(dev, b[6:12], task, b[14])
            msg = 'squeezing on the fixed device, continuous phase: %.2f%% %s' % (100 * (1 - out['sqfix_chk'][-1][0] / out['chk0'][-1][0]), dict(zip(KN, b[6:12].round(3))))
            if 'sqjoint' in out:
                bj = best(out['sqjoint']); dj = replace(dev, **dict(zip(rb.PN, bj[:6]))); out['sqjoint_chk'] = check(dj, bj[6:12], task, bj[14])
                msg += ' | + device %.2f%%' % (100 * (1 - out['sqjoint_chk'][-1][0] / out['chk0'][-1][0]))
            save(); log(name, msg)
        elif mode == 'sqfix2':
            # squeezing-type points on the fixed device found by any search (knob scans and refined knob search with g_cr = 0
            # exactly, first-order points, sqfix); the best is refined by a second Nelder-Mead (60 evaluations, small simplex)
            base_row = np.asarray(out['sqscan'])[0].copy(); base_row[6:12] = 0; base_row[12:15] = out['L0']
            T = np.vstack([base_row[None, :], first_rows(out['first'], out['L0'][0], base_row)] + [np.asarray(out[k]) for k in ('knob_trace', 'dec_kn', 'sqscan', 'sqfix') if k in out])
            T = T[T[:, 8] == 0]
            if 'sqfix2' not in out:
                ud = rb.to_u([dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps]); i0 = int(np.argmin(np.where(T[:, 14] <= NMAX, T[:, 12], np.inf)))
                out['sqfix2'] = masked_local(dev, task, np.concatenate([ud, kv(T[i0, 6:12])]), [6, 7, 10, 11], 60, step=0.03); save()
            b = best(np.vstack([T, out['sqfix2']])); out['sqfix_chk'] = check(dev, b[6:12], task, b[14]); out['sqfix_best'] = b; save()
            log(name, 'squeezing on the fixed device, continuous phase, best of all searches: %.2f%% (N_c=%d %.2f%%) %s nmax %.2f' % (
                100 * (1 - out['sqfix_chk'][-1][0] / out['chk0'][-1][0]), NC, 100 * (1 - b[12] / out['L0'][0]), dict(zip(KN, b[6:12].round(3))), b[14]))
        elif mode == 'ordctl':
            # control for sqjoint: the same local search over the six ordinary parameters with the knobs off
            if 'dec_ord' not in out:
                ud = rb.to_u([dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps])
                out['dec_ord'] = masked_local(dev, task, np.concatenate([ud, np.zeros(6) + kv(np.zeros(6))]), list(range(6)), 150); save()
            b = best(out['dec_ord']); dv = replace(dev, **dict(zip(rb.PN, b[:6]))); out['dec_ord_chk'] = check(dv, np.zeros(6), task, b[14]); save()
            log(name, 'ordinary-only local control: %.2f%%' % (100 * (1 - out['dec_ord_chk'][-1][0] / out['chk0'][-1][0])))
        elif mode == 'decomp':
            base_row = np.asarray(out['knob_trace'])[0].copy()
            KT = np.vstack([out['knob_trace'], first_rows(out['first'], out['L0'][0], base_row)])
            if 'dec_kn' not in out:          # knobs only, device fixed, Nelder-Mead from the best of scans and first-order points
                ud = rb.to_u([dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps]); pen = KT[:, 12]
                out['dec_kn'] = masked_local(dev, task, np.concatenate([ud, kv(KT[int(np.argmin(pen)), 6:12])]), list(range(6, 12)), 100, step=0.08); save()
            if 'dec_cr' not in out:
                D = decomp(dev, task, KT)
                for k, v in D.items(): out['dec_' + k] = v
                save()
            L0 = out['L0'][0]; res = {}
            for k in ('kn', 'ord', 'sq', 'cr'):
                b = best(out['dec_' + k]); dv = replace(dev, **dict(zip(rb.PN, b[:6]))); c = check(dv, b[6:12], task, b[14])
                res[k] = (b, c[-1][0]); out['dec_%s_chk' % k] = c
            save()
            c0 = out['chk0'][-1][0]
            log(name, 'knobs only, refined: %.2f%% %s' % (100 * (1 - res['kn'][1] / out['chk0'][-1][0]), dict(zip(KN, res['kn'][0][6:12].round(3)))))
            log(name, 'decomposition (checked, vs re-fabricated %.4e): ordinary only %.2f%%, + squeezing-type knobs %.2f%% %s, + counter-rotating only %.2f%% %s' % (
                c0, 100 * (1 - res['ord'][1] / c0), 100 * (1 - res['sq'][1] / c0), dict(zip(KN, res['sq'][0][6:12].round(2))),
                100 * (1 - res['cr'][1] / c0), dict(zip(KN, res['cr'][0][6:12].round(2)))))
        elif mode == 'path':
            b2 = base_device2(name, NC); pb = [b2.g, b2.kappa, b2.gamma, b2.omega, b2.omega_q, b2.eps]
            pr = [dev.g, dev.kappa, dev.gamma, dev.omega, dev.omega_q, dev.eps]; ub, ur = rb.to_u(pb), rb.to_u(pr)
            for s in (0.0, 1 / 3, 2 / 3):
                key = 'path_%d' % round(3 * s)
                if key in out: continue
                ds = replace(dev, **dict(zip(rb.PN, rb.to_p(ub + s * (ur - ub)))))
                L0 = evaluate_ps(ds, np.zeros(6), task); T = knob_search(ds, task, 120); bk = best(T)
                out[key] = np.array([s, L0[0], bk[12], *bk[6:12]]); out[key + '_trace'] = T; save()
                log(name, 'path s = %.2f: ordinary L %.4e, with knobs %.4e (%.1f%%)' % (s, L0[0], bk[12], 100 * (1 - bk[12] / L0[0])), dict(zip(KN, bk[6:12].round(3))))
    log('done', sys.argv[1:])
