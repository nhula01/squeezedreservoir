"""Spoken digits, base-first: (1) drive amplitude optimized at zero squeezing on fold 1 (photon budget 2.5),
(2) 5-fold WER at that base, (3) (r, r_e) grids at phi_d in {0, pi/2} from the base (fold-1 training loss, budget-masked),
(4) 5-fold WER at the best squeezed point selected by fold-1 training loss."""
from run_asr import *
NMAX = 2.5
if __name__ == '__main__':
    fp = os.path.join(DATA, 'asr_base.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    d = np.load(os.path.join(DATA, 'asr_features.npz')); Xm, Xs, y = d['Xm'], d['Xs'], d['y']; FOLDS = folds(); tr0, te0 = FOLDS[0]
    if 'eps' not in out:
        rows = []
        for e in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            Nc = 8 if e <= 0.3 else 10 if e <= 0.5 else 12
            r = asr_fold(replace(DEV, eps=e, Nc=Nc), (0, 0, 0), Xm, Xs, y, tr0, te0); rows.append([e, *r]); log('eps %.2f L %.5f WER %.2f nmax %.2f' % (e, r[0], r[1], r[3]))
        rows = np.array(rows); ok = rows[:, 4] <= NMAX; eb = rows[ok][rows[ok][:, 1].argmin(), 0]; out['eps'], out['epsbest'] = rows, eb; np.savez(fp, **out)
    eb = float(out['epsbest']); Nc = 8 if eb <= 0.3 else 10 if eb <= 0.5 else 12; dev = replace(DEV, eps=eb, Nc=max(Nc, 10))
    if 'base5' not in out:
        r = [asr_fold(dev, (0, 0, 0), Xm, Xs, y, tr, te) for tr, te in FOLDS]; out['base5'] = np.array(r); log('base eps %.2f 5-fold WER' % eb, np.array(r)[:, 1], np.mean([x[1] for x in r])); np.savez(fp, **out)
    grid = np.linspace(0, 0.5, 4)
    for pd, tag in ((0.0, 'p'), (np.pi / 2, 'm')):
        if f'grid_{tag}' in out: continue
        G = np.zeros((4, 4, 4))
        for i, r_ in enumerate(grid):
            for j, re_ in enumerate(grid): G[i, j] = asr_fold(replace(dev, phi_d=pd, Nc=12), (r_, re_, 0.0), Xm, Xs, y, tr0, te0)
            log(tag, 'r', r_, 'L', G[i, :, 0].round(5), 'WER', G[i, :, 1].round(2), 'nmax', G[i, :, 3].round(2))
        G[:, :, 0] = np.where(G[:, :, 3] <= NMAX, G[:, :, 0], np.inf); out[f'grid_{tag}'] = G; out['grid'] = grid; np.savez(fp, **out)
    if 'best5' not in out:
        cands = [(out[f'grid_{t}'][:, :, 0].min(), pd, t) for pd, t in ((0.0, 'p'), (np.pi / 2, 'm'))]; Lb, pd, t = min(cands)
        i, j = np.unravel_index(out[f'grid_{t}'][:, :, 0].argmin(), (4, 4)); th = (grid[i], grid[j], 0.0)
        r = [asr_fold(replace(dev, phi_d=pd, Nc=12), th, Xm, Xs, y, tr, te) for tr, te in FOLDS]; out['best5'] = np.array(r); out['best'] = np.array(th); out['bestpd'] = pd
        log('squeezed on base: theta', th, 'phi_d %.2f' % pd, '5-fold WER', np.array(r)[:, 1], np.mean([x[1] for x in r])); np.savez(fp, **out)
    log('done')
