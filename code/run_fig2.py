"""Fig. 2: task-loss landscapes over the squeezing controls + gradient trajectories + truncation mask."""
from common import *
rng = np.random.default_rng(2)
def scan(grid_r, grid_a, grid_b, fixed, which, dev=DEV, task=MG):
    """which='r_dth': grid over (r, dth) at re=fixed ; which='r_re': grid over (r, re) at dth=fixed."""
    L = np.zeros((len(grid_a), len(grid_b))); NR = L.copy(); NM = L.copy()
    for i, a in enumerate(grid_a):
        for j, b in enumerate(grid_b):
            th = (a, fixed, b) if which == 'r_dth' else (a, b, fixed)
            L[i, j], NR[i, j], NM[i, j] = evaluate(dev, th, task)
        log(which, i, '/', len(grid_a), 'row min', L[i].min())
    return L, NR, NM

if __name__ == '__main__':
    out = {}
    # coarse 3D scan
    r3, re3, d3 = np.linspace(0, 0.5, 6), np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 13)[:-1]
    L3 = np.zeros((6, 6, 12)); N3 = L3.copy(); M3 = L3.copy()
    for i, r in enumerate(r3):
        for j, re in enumerate(re3):
            for k, d in enumerate(d3):
                L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(DEV, (r, re, d), MG)
        log('3D scan', i)
    i, j, k = np.unravel_index(L3.argmin(), L3.shape)
    log('coarse best', r3[i], re3[j], d3[k], L3.min(), 'unsqueezed', L3[0, 0, 0])
    out.update(r3=r3, re3=re3, d3=d3, L3=L3, N3=N3, M3=M3)
    re_fix = re3[j]
    # dense slice 1: (r, dth) at re = re_fix
    n = 25
    rg, dg = np.linspace(0, 0.5, n), np.linspace(-np.pi, np.pi, n)
    L1, NR1, NM1 = scan(None, rg, dg, re_fix, 'r_dth')
    i1, j1 = np.unravel_index(L1.argmin(), L1.shape)
    dth_fix = dg[j1]
    log('slice1 best', rg[i1], re_fix, dth_fix, L1.min())
    out.update(rg=rg, dg=dg, re_fix=re_fix, L1=L1, NR1=NR1, NM1=NM1)
    # dense slice 2: (r, re) at dth = dth_fix
    reg = np.linspace(0, 0.5, n)
    L2, NR2, NM2 = scan(None, rg, reg, dth_fix, 'r_re')
    i2, j2 = np.unravel_index(L2.argmin(), L2.shape)
    log('slice2 best', rg[i2], reg[j2], dth_fix, L2.min())
    out.update(reg=reg, dth_fix=dth_fix, L2=L2, NR2=NR2, NM2=NM2)
    np.savez(os.path.join(DATA, 'fig2_scans.npz'), **out)
    # truncation mask on coarse 7x7 subgrids of both slices: Nc=8 vs Nc=12
    dev12 = replace(DEV, Nc=12)
    rc, dc, rec = np.linspace(0, 0.5, 7), np.linspace(-np.pi, np.pi, 7), np.linspace(0, 0.5, 7)
    T1 = np.zeros((7, 7)); T2 = np.zeros((7, 7))
    for i, r in enumerate(rc):
        for j, d in enumerate(dc):
            a = evaluate(DEV, (r, re_fix, d), MG)[0]; b = evaluate(dev12, (r, re_fix, d), MG)[0]
            T1[i, j] = abs(a - b) / b
        for j, re in enumerate(rec):
            a = evaluate(DEV, (r, re, dth_fix), MG)[0]; b = evaluate(dev12, (r, re, dth_fix), MG)[0]
            T2[i, j] = abs(a - b) / b
        log('trunc row', i)
    out.update(rc=rc, dc=dc, rec=rec, T1=T1, T2=T2)
    np.savez(os.path.join(DATA, 'fig2_scans.npz'), **out)
    # gradient trajectories inside each slice (2D projected GD), 4 random starts, 60 evaluations each
    paths1, paths2 = [], []
    for s in range(4):
        th0 = np.array([rng.uniform(0, 0.5), re_fix, rng.uniform(-np.pi, np.pi)])
        cnt = Counter(DEV, MG, budget=60)
        p = gradient_descent(cnt, th0, 60, active=(0, 2))
        paths1.append(np.array([[*t, L] for t, L, _ in p]))
        log('traj1', s, 'evals', cnt.n, 'L', p[0][1], '->', p[-1][1])
        th0 = np.array([rng.uniform(0, 0.5), rng.uniform(0, 0.5), dth_fix])
        cnt = Counter(DEV, MG, budget=60)
        p = gradient_descent(cnt, th0, 60, active=(0, 1))
        paths2.append(np.array([[*t, L] for t, L, _ in p]))
        log('traj2', s, 'evals', cnt.n, 'L', p[0][1], '->', p[-1][1])
    np.savez(os.path.join(DATA, 'fig2_paths.npz'), paths1=np.array(paths1, dtype=object), paths2=np.array(paths2, dtype=object), allow_pickle=True)
    log('done fig2')
