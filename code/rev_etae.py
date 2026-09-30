"""Revision 2: sensitivity of the measurement cost (Sec. 2.6, rev_post.py) to the emitter-fluorescence detection
efficiency, eta_e = 0.1, 0.3, 0.5 (eta_c = 0.8), at the tuned base and its squeezed optimum of every task.
Rows per (task, point, eta_e): [N_rep, median sigma_rel, sigma_rel per feature class (Q,P,Q2,P2,sx,sy), mean L, std L,
mean NRMSE, std NRMSE] over 12 readout-noise draws. Output: data/rev_etae.npz"""
from rev_common import *
from rev_calib import NREP
ETAS = (0.1, 0.3, 0.5)
if __name__ == '__main__':
    fp = os.path.join(DATA, 'rev_etae.npz'); out = {}; rng = np.random.default_rng(12); ndraw = 12
    for n, task in TASKS.items():
        dsq, th = squeezed_point2(n)
        for k, t in (('base', np.zeros(3)), ('sq', th)):
            F, E, _ = ResX(dsq, t).run_moments(task.f); L0 = evaluate(dsq, t, task, feats=F); out[f'{n}_{k}_L0'] = np.array(L0[:2])
            for eta in ETAS:
                cov = single_run_cov(dsq, F, E, 0.8, eta); rows = []
                for N in NREP:
                    srel, rel = realized_sigma_rel(dsq, task, F, cov, N)
                    a = np.array([evaluate(dsq, t, task, feats=add_readout_noise(dsq, F, cov, N, rng))[:2] for _ in range(ndraw)])
                    rows.append([N, srel, *[np.median(rel[i::NOBS]) for i in range(NOBS)], a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std()])
                out[f'{n}_{k}_eta{eta}'] = np.array(rows)
        for eta in ETAS:
            rb, rs = out[f'{n}_base_eta{eta}'], out[f'{n}_sq_eta{eta}']
            log(n, 'eta_e %.1f gain vs N_rep:' % eta, ' '.join('%.0e:%.0f%%' % (N, 100 * (1 - rs[i, 8] / rb[i, 8])) for i, N in enumerate(NREP)),
                'clean %.0f%%' % (100 * (1 - out[f'{n}_sq_L0'][0] / out[f'{n}_base_L0'][0])), 'srel(1e6) sq %.3f' % rs[3, 1])
        np.savez(fp, NREP=NREP, ETAS=np.array(ETAS), **out)
    log('done etae')
