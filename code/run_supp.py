"""Supplementary: (i) finite-difference step convergence, (ii) measurement-noise gradients + noisy GD,
(iii) NARMA10 coarse landscape, (iv) Fock-cutoff convergence at the best points, (v) ridge sensitivity."""
from common import *
from run_fig3 import run as run_opt

def anchors():
    sc = np.load(os.path.join(DATA, 'fig2_scans.npz'))
    i, j, k = np.unravel_index(sc['L3'].argmin(), sc['L3'].shape)
    best = np.array([sc['r3'][i], sc['re3'][j], sc['d3'][k]])
    mid = np.array([0.25, 0.25, 1.0])
    best[0] = max(best[0], 0.02)                     # keep the anchor interior so central differences stay two-sided
    return {'unsqueezed': np.array([0.02, 0.02, 0.0]), 'mid': mid, 'basin': best}

if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = {}
    A = anchors()
    if part in ('fd', 'all'):
        hs = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        for name, th in A.items():
            G = np.zeros((len(hs), 3))
            for a, h in enumerate(hs):
                cnt = Counter(DEV, MG)
                G[a] = fd_gradient(cnt, th, h=np.array([h, h, h]))
                log('fd', name, h, G[a])
            out[f'fd_{name}'] = G; out[f'anchor_{name}'] = th
        out['fd_h'] = hs
        np.savez(os.path.join(DATA, 'supp_fd.npz'), **out)
    if part in ('noise', 'noisegrad', 'all'):
        # gradients from noisy features: store features at theta +- h e_i for several h, then add noise
        out = {}
        hs = np.array([1e-2, 3e-2, 1e-1])
        sig = np.array([1e-3, 3e-3, 1e-2, 3e-2])
        nrep = 100
        for name, th in [('mid', A['mid']), ('basin', A['basin'])]:
            feats = {}
            for a, h in enumerate(hs):
                for i in range(3):
                    for sgn in (+1, -1):
                        e = np.zeros(3); e[i] = sgn * h
                        feats[(a, i, sgn)] = Reservoir(DEV, project(th + e)).run(MG.f)[0]
            # noiseless
            G0 = np.zeros((len(hs), 3))
            Gn = np.zeros((len(hs), len(sig), nrep, 3))
            rng = np.random.default_rng(7)
            for a, h in enumerate(hs):
                for i in range(3):
                    Lp = evaluate(DEV, th, MG, feats=feats[(a, i, 1)])[0]; Lm = evaluate(DEV, th, MG, feats=feats[(a, i, -1)])[0]
                    G0[a, i] = (Lp - Lm) / (2 * h)
                    for b, s in enumerate(sig):
                        for r_ in range(nrep):
                            Lp = evaluate(DEV, th, MG, feats=feats[(a, i, 1)], noise=s, rng=rng)[0]
                            Lm = evaluate(DEV, th, MG, feats=feats[(a, i, -1)], noise=s, rng=rng)[0]
                            Gn[a, b, r_, i] = (Lp - Lm) / (2 * h)
            out[f'G0_{name}'] = G0; out[f'Gn_{name}'] = Gn; out[f'anchor_{name}'] = th
            log('noise', name, 'G0', G0)
        out['hs'] = hs; out['sig'] = sig
        np.savez(os.path.join(DATA, 'supp_noise.npz'), **out)
    if part in ('noise', 'all'):
        # noisy gradient descent: fresh noise on every evaluation, larger FD step
        rng = np.random.default_rng(11)
        inits = random_init(rng, 4)
        res = {}
        for s, h in ((3e-3, 3e-2), (1e-2, 1e-1)):
            curves = np.zeros((len(inits), 100)); clean = np.zeros(len(inits))
            for k, th0 in enumerate(inits):
                cnt = Counter(DEV, MG, noise=s, rng=np.random.default_rng(100 + k), budget=100)
                p = gradient_descent(cnt, th0, 100, h=np.array([h, h, h]))
                curves[k] = cnt.best_so_far(100)
                # noiseless loss at the final accepted point
                clean[k] = evaluate(DEV, p[-1][0], MG)[0]
                log('noisyGD', s, k, 'L0=%.4e noisy_best=%.4e clean_final=%.4e' % (curves[k][0], curves[k][-1], clean[k]))
            res[f'curves_{s}'] = curves; res[f'clean_{s}'] = clean
            res['L0_clean'] = np.array([evaluate(DEV, th0, MG)[0] for th0 in inits])
            np.savez(os.path.join(DATA, 'supp_noisygd.npz'), inits=inits, **res)
    if part in ('narma', 'all'):
        if not os.path.exists(os.path.join(DATA, 'supp_narma_scan.npz')):
            r3, re3, d3 = np.linspace(0, 0.5, 6), np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 13)[:-1]
            L3 = np.zeros((6, 6, 12)); N3 = L3.copy(); M3 = L3.copy()
            for i, r in enumerate(r3):
                for j, re in enumerate(re3):
                    for k, d in enumerate(d3):
                        L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(DEV, (r, re, d), NARMA)
                log('narma scan', i, L3[i].min())
            np.savez(os.path.join(DATA, 'supp_narma_scan.npz'), r3=r3, re3=re3, d3=d3, L3=L3, N3=N3, M3=M3)
        run_opt(NARMA, DEV, n_init=4, budget=60, seed=4, tag='narma')
    if part in ('fock', 'all'):
        out = {}
        pts = dict(A)
        try:
            f3 = np.load(os.path.join(DATA, 'fig3_mg.npz'), allow_pickle=True)
            fg = f3['final_gd']; pts['best_gd'] = fg[fg[:, 3].argmin(), :3]
        except Exception as e:
            log('no fig3 yet', e)
        pts['corner'] = np.array([0.5, 0.5, 0.0])
        Ncs = [6, 8, 10, 12, 16, 20]
        for name, th in pts.items():
            rows = []
            for Nc in Ncs:
                L, nr, nm = evaluate(replace(DEV, Nc=Nc), th, MG)
                rows.append([Nc, L, nr, nm]); log('fock', name, Nc, L, nm)
            out[f'fock_{name}'] = np.array(rows); out[f'anchor_{name}'] = th
        np.savez(os.path.join(DATA, 'supp_fock.npz'), **out)
    if part in ('ridge', 'all'):
        # ridge sensitivity: recompute the coarse landscape from stored features at several sigma_rel
        r3, re3, d3 = np.linspace(0, 0.5, 6), np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 13)[:-1]
        sigs = [0.003, 0.01, 0.03, 0.1]
        L = np.zeros((len(sigs), 6, 6, 12)); NR = L.copy()
        for i, r in enumerate(r3):
            for j, re in enumerate(re3):
                for k, d in enumerate(d3):
                    F = Reservoir(DEV, (r, re, d)).run(MG.f)[0]
                    for a, s in enumerate(sigs):
                        L[a, i, j, k], NR[a, i, j, k], _ = evaluate(replace(DEV, sigma_rel=s), (r, re, d), MG, feats=F)
            log('ridge scan', i)
        np.savez(os.path.join(DATA, 'supp_ridge.npz'), sigs=np.array(sigs), L=L, NR=NR, r3=r3, re3=re3, d3=d3)
    log('done supp', part)
