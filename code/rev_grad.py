"""Concern 2a: finite-difference gradient estimation under the readout model, on the revised (drive amplitude and
frequency optimized) base device of the task with the largest squeezing gain (rev_common.training_task).
At two interior anchors: clean (Richardson) gradient, third-derivative estimate, loss-estimator noise sigma_L(N_rep), and
the error / sign recovery of central-difference gradients from noisy features for h in H and N_rep in NREP."""
from rev_common import *
NC = 10
H = np.array([0.01, 0.03, 0.1]); NREP = np.array([1e5, 1e6, 1e7]); NDRAW = 40
ANCHORS = {'start': np.array([0.05, 0.05, 0.0]), 'mid': np.array([0.25, 0.25, 1.0])}
if __name__ == '__main__':
    NAME = training_task(); task = TASKS[NAME]; dev, _ = squeezed_point2(NAME, NC); rng = np.random.default_rng(17); out = {}
    for an, th in ANCHORS.items():
        clean = lambda t: evaluate(dev, t, task)[0]
        g_true, g_err, _ = fd_gradient_conv(clean, th, tol=0.02, return_info=True)
        F0, E0, _ = ResX(dev, th).run_moments(task.f); cov0 = single_run_cov(dev, F0, E0)
        sigL = np.array([np.std([evaluate(dev, th, task, feats=add_readout_noise(dev, F0, cov0, N, rng))[0] for _ in range(NDRAW)]) for N in NREP])
        meanL = np.array([np.mean([evaluate(dev, th, task, feats=add_readout_noise(dev, F0, cov0, N, rng))[0] for _ in range(NDRAW)]) for N in NREP])
        Dclean = np.zeros((len(H), 3)); G = np.zeros((len(H), len(NREP), NDRAW, 3))
        for a, h in enumerate(H):
            for i in range(3):
                FF = {}
                for sg in (1, -1):
                    e = np.zeros(3); e[i] = sg * h; tp = th + e
                    F, E, _ = ResX(dev, tp).run_moments(task.f); FF[sg] = (tp, F, single_run_cov(dev, F, E))
                Dclean[a, i] = (evaluate(dev, FF[1][0], task, feats=FF[1][1])[0] - evaluate(dev, FF[-1][0], task, feats=FF[-1][1])[0]) / (2 * h)
                for b, N in enumerate(NREP):
                    for d in range(NDRAW):
                        Lp = evaluate(dev, FF[1][0], task, feats=add_readout_noise(dev, FF[1][1], FF[1][2], N, rng))[0]
                        Lm = evaluate(dev, FF[-1][0], task, feats=add_readout_noise(dev, FF[-1][1], FF[-1][2], N, rng))[0]
                        G[a, b, d, i] = (Lp - Lm) / (2 * h)
            log(an, 'h', h, 'clean D', Dclean[a].round(5), 'true', g_true.round(5))
        c3 = (Dclean[1] - Dclean[0]) / (H[1] ** 2 - H[0] ** 2)          # D(h) = g + (L'''/6) h^2
        out[f'{an}_theta'] = th; out[f'{an}_gtrue'] = g_true; out[f'{an}_gerr'] = g_err; out[f'{an}_sigL'] = sigL; out[f'{an}_meanL'] = meanL
        out[f'{an}_L0'] = clean(th); out[f'{an}_Dclean'] = Dclean; out[f'{an}_G'] = G; out[f'{an}_L3'] = 6 * c3
        log(an, 'sigma_L', sigL, 'L0', out[f'{an}_L0'], "L'''", 6 * c3)
    np.savez(os.path.join(DATA, 'rev_grad.npz'), task=NAME, H=H, NREP=NREP, **out)
