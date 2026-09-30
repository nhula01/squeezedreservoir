"""Fig. 4: feature sensitivity S_i, task-gradient magnitude G_i and diagonal QFI along lines in control space,
plus the final loss reached by gradient descent initialized along the line."""
from common import *

def qfi_diag(rho, drho, tol=1e-12):
    lam, V = np.linalg.eigh(rho)
    d = V.conj().T @ drho @ V
    s = lam[:, None] + lam[None, :]
    mask = s > tol * lam.max()
    return 2 * np.sum(np.abs(d[mask]) ** 2 / s[mask])

def line_point(dev, task, theta, h=H_FD):
    """Return S (3,), G (3,), QFI (3,) [per-sample mean over training window], L, nrmse, nmax, and Bq-like sum."""
    res0 = Reservoir(dev, theta)
    F0, nmax, S0 = res0.run(task.f, return_states=True)
    tr, te = task.split()
    mu, sd = F0[tr].mean(0), F0[tr].std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    L0, nr0, _ = evaluate(dev, theta, task, feats=F0)
    rhos = [res0.to_matrix(c) for c in S0]
    S, G, Q = np.zeros(3), np.zeros(3), np.zeros(3)
    Sg = np.zeros((3, 3))                                     # per observable group: (Q,P), (Q^2,P^2), (sx,sy)
    grp = (np.arange(NOBS * dev.Nv) % NOBS) // 2
    for i in range(3):
        e = np.zeros(3); e[i] = h[i]
        rp = Reservoir(dev, project(theta + e)); Fp, _, Sp = rp.run(task.f, return_states=True)
        rm = Reservoir(dev, project(theta - e)); Fm, _, Sm = rm.run(task.f, return_states=True)
        dX = (Fp - Fm)[tr] / (2 * h[i]) / sd
        dX = dX - dX.mean(0)                                   # a bias column absorbs a uniform shift
        S[i] = np.linalg.norm(dX) / np.sqrt(len(dX))          # per-sample RMS feature response
        for gi in range(3):
            Sg[i, gi] = np.linalg.norm(dX[:, grp == gi]) / np.sqrt(len(dX))
        Lp = evaluate(dev, theta + e, task, feats=Fp)[0]; Lm = evaluate(dev, theta - e, task, feats=Fm)[0]
        G[i] = abs(Lp - Lm) / (2 * h[i])
        q = 0.0
        for t in range(tr.start, tr.stop):
            drho = (rp.to_matrix(Sp[t]) - rm.to_matrix(Sm[t])) / (2 * h[i])
            q += qfi_diag(rhos[t], drho)
        Q[i] = q / (tr.stop - tr.start)
    return S, G, Q, L0, nr0, nmax, Sg

if __name__ == '__main__':
    sc = np.load(os.path.join(DATA, 'fig2_scans.npz'))
    re_fix, dth_fix = float(sc['re_fix']), float(sc['dth_fix'])
    n = 21
    lines = {'r': [(r, re_fix, dth_fix) for r in np.linspace(0, 0.5, n)],
             're': [(0.0, re, dth_fix) for re in np.linspace(0, 0.5, n)],
             'dth': [(0.0, re_fix, d) for d in np.linspace(-np.pi, np.pi, n)]}
    out = dict(re_fix=re_fix, dth_fix=dth_fix)
    fp = os.path.join(DATA, 'fig4_lines.npz')
    if os.path.exists(fp):
        out.update({k: v for k, v in np.load(fp).items()})
    for name, pts in lines.items():
        if f'line_{name}' in out:
            continue
        arr = np.zeros((n, 3 + 3 + 3 + 3 + 9))
        for k, th in enumerate(pts):
            S, G, Q, L, nr, nm, Sg = line_point(DEV, MG, np.array(th))
            arr[k] = [*S, *G, *Q, L, nr, nm, *Sg.ravel()]
            log('line', name, k, 'L=%.4e' % L, 'S', np.round(S, 3), 'G', np.round(G, 5), 'Q', np.round(Q, 3))
        out[f'line_{name}'] = arr; out[f'pts_{name}'] = np.array(pts)
        np.savez(os.path.join(DATA, 'fig4_lines.npz'), **out)
    # final loss reached by full 3D gradient descent initialized along the r line (7 starts, 60 evaluations)
    starts = [(r, re_fix, dth_fix) for r in np.linspace(0, 0.5, 7)]
    fin = []
    for th0 in ([] if 'init_r' in out else starts):
        cnt = Counter(DEV, MG, budget=60)
        p = gradient_descent(cnt, np.array(th0), 60)
        tr = np.array(cnt.trace); ib = tr[:, 4].argmin()
        fin.append([*th0, p[0][1], tr[ib, 4], *tr[ib, 1:4]])
        log('init', np.round(th0, 3), 'L0=%.4e -> %.4e' % (p[0][1], tr[ib, 4]))
    if fin: out['init_r'] = np.array(fin); np.savez(fp, **out)
    # and along the re line at r = 0
    starts = [(0.0, re, dth_fix) for re in np.linspace(0, 0.5, 7)]
    fin = []
    for th0 in ([] if 'init_re' in out else starts):
        cnt = Counter(DEV, MG, budget=60)
        p = gradient_descent(cnt, np.array(th0), 60)
        tr = np.array(cnt.trace); ib = tr[:, 4].argmin()
        fin.append([*th0, p[0][1], tr[ib, 4], *tr[ib, 1:4]])
        log('init', np.round(th0, 3), 'L0=%.4e -> %.4e' % (p[0][1], tr[ib, 4]))
    if fin: out['init_re'] = np.array(fin)
    np.savez(fp, **out)
    log('done fig4')
