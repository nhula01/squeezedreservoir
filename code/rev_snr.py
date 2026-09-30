"""SNR-aware readout: measured squeezing gain (training loss and held-out NRMSE) against N_rep with the protocol readout
and with the SNR-aware readout (expt.evaluate_snr), at the tuned base and its squeezed optimum of every task; and the
loss-estimator noise against N_rep at the channel-equalization gradient anchors, for both readouts."""
from rev_common import *
from rev_calib import NREP
if __name__ == '__main__':
    out = {}; rng = np.random.default_rng(8); ndraw = 12
    for n, task in TASKS.items():
        dsq, th = squeezed_point2(n)
        for k, t in (('base', np.zeros(3)), ('sq', th)):
            F, E, _ = ResX(dsq, t).run_moments(task.f); cov = single_run_cov(dsq, F, E); rows = []
            for N in NREP:
                nv = noise_var_vector(dsq, task, cov, N); a = []; b = []
                for _ in range(ndraw):
                    Fn = add_readout_noise(dsq, F, cov, N, rng)
                    a.append(evaluate(dsq, t, task, feats=Fn)[:2]); b.append(evaluate_snr(dsq, task, Fn, nv))
                a, b = np.array(a), np.array(b)
                tr_, _ = task.split(); keep = (nv < SNR_THRESH * F[tr_].var(0)).mean()
                rows.append([N, *a.mean(0), *b.mean(0), keep])
            out[f'{n}_{k}'] = np.array(rows); out[f'{n}_{k}_clean'] = np.array(evaluate_nm(dsq, task, F))
        rb, rs = out[f'{n}_base'], out[f'{n}_sq']
        log(n, 'protocol', ' '.join('%.0e:%.0f' % (N, 100 * (1 - rs[i, 1] / rb[i, 1])) for i, N in enumerate(NREP)),
            '| snr', ' '.join('%.0e:%.0f' % (N, 100 * (1 - rs[i, 3] / rb[i, 3])) for i, N in enumerate(NREP)),
            '| snr heldout', ' '.join('%.0e:%.0f' % (N, 100 * (1 - rs[i, 4] / rb[i, 4])) for i, N in enumerate(NREP)), '| kept', np.round(rs[:, 5], 2))
    dev, _ = squeezed_point2('nce'); task = TASKS['nce']
    for an, th in (('start', np.array([0.05, 0.05, 0.0])), ('mid', np.array([0.25, 0.25, 1.0]))):
        F, E, _ = ResX(dev, th).run_moments(task.f); cov = single_run_cov(dev, F, E); s = []
        for N in (1e5, 1e6, 1e7):
            nv = noise_var_vector(dev, task, cov, N)
            s.append([np.std([evaluate(dev, th, task, feats=add_readout_noise(dev, F, cov, N, rng))[0] for _ in range(40)]),
                      np.std([evaluate_snr(dev, task, add_readout_noise(dev, F, cov, N, rng), nv)[0] for _ in range(40)]),
                      evaluate(dev, th, task, feats=F)[0], evaluate_nm(dev, task, F)[0]])
        out[f'sigL_{an}'] = np.array(s); log('sigma_L', an, np.array(s).round(5))
    np.savez(os.path.join(DATA, 'rev_snr.npz'), NREP=NREP, **out)
