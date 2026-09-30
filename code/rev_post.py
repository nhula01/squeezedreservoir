"""After rev_base2: (i) readout-noise calibration at the revised base and its squeezed optimum for every task;
(ii) mapped effective device for the revised squeezed optima."""
from rev_common import *
from rev_refab import mapped as _mapped
from rev_calib import NREP
if __name__ == '__main__':
    fp = os.path.join(DATA, 'rev_post.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    rng = np.random.default_rng(6); ndraw = 12
    for n, task in TASKS.items():
        if f'{n}_sq' in out: continue
        dsq, th = squeezed_point2(n); pts = {'base': (dsq, np.zeros(3)), 'sq': (dsq, th)}
        for k, (dv, t) in pts.items():
            F, E, nm = ResX(dv, t).run_moments(task.f); cov = single_run_cov(dv, F, E); tr, te = task.split()
            L0 = evaluate(dv, t, task, feats=F); rows = []
            for N in NREP:
                srel, rel = realized_sigma_rel(dv, task, F, cov, N)
                a = [evaluate(dv, t, task, feats=add_readout_noise(dv, F, cov, N, rng))[:2] for _ in range(ndraw)]
                a = np.array(a); rows.append([N, srel, *[np.median(rel[i::NOBS]) for i in range(NOBS)], a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std()])
            out[f'{n}_{k}'] = np.array(rows); out[f'{n}_{k}_L0'] = np.array(L0[:2])
        rb, rs = out[f'{n}_base'], out[f'{n}_sq']
        log(n, 'gain vs N', ' '.join('%.0e:%.0f%%' % (N, 100 * (1 - rs[i, 8] / rb[i, 8])) for i, N in enumerate(NREP)), 'clean %.0f%%' % (100 * (1 - out[f'{n}_sq_L0'][0] / out[f'{n}_base_L0'][0])))
        np.savez(fp, NREP=NREP, **out)
    # mapped effective device at the revised optima
    for n, task in TASKS.items():
        if f'{n}_map2_Lfull' in out: continue
        dsq, th = squeezed_point2(n)
        e = effective_parameters(dsq, th)
        lin = replace(dsq, omega=e['Omega'], g=e['g_co'], eps=e['eps_eff'], phi_d=0.0)
        Llin = evaluate(lin, (0, 0, 0), task)
        target = np.mean([e['gamma_x'], e['gamma_y'], e['gamma_z']])
        gm = lambda x: np.mean([effective_parameters(replace(lin, gamma=x), (0, 0, 0))[k] for k in ('gamma_x', 'gamma_y', 'gamma_z')])
        lo, hi = 1e-3, 2.0
        if gm(lo) > target: gam = lo
        else:
            for _ in range(16):
                mid = np.sqrt(lo * hi); lo, hi = (mid, hi) if gm(mid) < target else (lo, mid)
            gam = np.sqrt(lo * hi)
        Lfull = evaluate(replace(lin, gamma=gam), (0, 0, 0), task)
        out[f'{n}_map2_eff'] = np.array([[k, v] for k, v in e.items()], dtype=object)
        out[f'{n}_map2_Llin'] = np.array(Llin); out[f'{n}_map2_Lfull'] = np.array(Lfull); out[f'{n}_map2_gamma'] = gam
        log(n, 'map2: lin %.4e full %.4e (gamma %.3f)' % (Llin[0], Lfull[0], gam)); np.savez(fp, NREP=NREP, **out)
    log('done post')
