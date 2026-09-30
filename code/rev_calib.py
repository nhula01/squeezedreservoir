"""Readout-noise calibration: realized feature precision and training loss against passes per setting (N_rep),
base (drive-optimized, unsqueezed) and squeezed optimum of every task, protocol ridge and noise-matched ridge."""
from rev_common import *
NREP = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
if __name__ == '__main__':
    fp = os.path.join(DATA, 'rev_calib.npz'); out = {}; rng = np.random.default_rng(5); ndraw = 12
    for n, task in TASKS.items():
        dsq, th = squeezed_point(n); pts = {'base': (replace(base_device(n), phi_d=dsq.phi_d, Nc=10), np.zeros(3)), 'sq': (replace(dsq, Nc=10), th)}
        for k, (dv, t) in pts.items():
            F, E, nm = ResX(dv, t).run_moments(task.f); cov = single_run_cov(dv, F, E)
            L0 = evaluate(dv, t, task, feats=F)[0]; rows = []
            for N in NREP:
                srel, rel = realized_sigma_rel(dv, task, F, cov, N); nv = noise_var_vector(dv, task, cov, N)
                a = []; b = []
                for _ in range(ndraw):
                    Fn = add_readout_noise(dv, F, cov, N, rng)
                    a.append(evaluate(dv, t, task, feats=Fn)[0]); b.append(evaluate_nm(dv, task, Fn, nv)[0])
                cls = [np.median(rel[i::NOBS]) for i in range(NOBS)]
                rows.append([N, srel, *cls, np.mean(a), np.std(a), np.mean(b), np.std(b)])
            out[f'{n}_{k}'] = np.array(rows); out[f'{n}_{k}_L0'] = L0
            out[f'{n}_{k}_L0nm'] = evaluate_nm(dv, task, F)[0]
            log(n, k, 'clean %.4e' % L0, ' '.join('N=%.0e:%.3f/%.3f' % (r[0], r[-4] / L0, r[-2] / L0) for r in rows), 'srel', np.round(np.array(rows)[:, 1], 3))
        np.savez(fp, NREP=NREP, **out)
