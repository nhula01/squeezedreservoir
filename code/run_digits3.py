"""Spoken digits: joint (r, r_e) scan at fixed phase on both parametric branches, plus a feature-motion diagnostic."""
from run_digits2 import *
if __name__ == '__main__':
    fp = os.path.join(DATA, 'digits_rre.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    items = load_subset(); f, frames = build_sequence(items)
    rng = np.random.default_rng(1); utts = np.arange(len(items)); digits = np.array([it[0] for it in items]); train_utts, test_utts = [], []
    for d in range(10):
        u = utts[digits == d]; rng.shuffle(u); train_utts += list(u[:len(u) // 2]); test_utts += list(u[len(u) // 2:])
    keys = ['cv_utt', 'cv_ce', 'cv_acc', 'delta', 'train_utt', 'train_acc', 'test_utt', 'test_ce', 'test_acc', 'frame_mse', 'nmax']
    dw = np.load(os.path.join(DATA, 'digits_wta.npz'), allow_pickle=True); S = dw['scan']; j, k = np.unravel_index(S[:, :, 0].argmin(), S[:, :, 0].shape); dth = float(dw['dg'][k])
    log('fixed phase', dth, 'from scan minimum at re', dw['reg'][j])
    grid = np.linspace(0, 0.5, 6)
    # feature-motion diagnostic: relative change of the standardized utterance-mean features between settings
    if 'motion' not in out:
        F0, _ = Reservoir(DEV, (0, 0, 0)).run(f); mot = []
        for th, s in (((0.5, 0.0, dth), 1.0), ((0.0, 0.5, dth), 1.0), ((0.5, 0.5, dth), 1.0), ((0.5, 0.5, dth), -1.0), ((0.25, 0.25, dth), 1.0)):
            F1, _ = Reservoir(replace(DEV, s=s), th).run(f)
            X0 = np.array([F0[idx].ravel() for (u, d, idx) in frames]); X1 = np.array([F1[idx].ravel() for (u, d, idx) in frames]); sd = X0.std(0) + 1e-12
            mot.append([*th, s, np.sqrt(np.mean(((X1 - X0) / sd) ** 2))]); log('feature motion (rms, in units of feature std)', mot[-1])
        out['motion'] = np.array(mot); np.savez(fp, **out)
    for s, tag in ((1.0, 'p'), (-1.0, 'm')):
        if f'grid_{tag}' in out: continue
        G = np.zeros((6, 6, len(keys)))
        for i, r_ in enumerate(grid):
            for jj, re_ in enumerate(grid):
                rr = wta_task(replace(DEV, s=s), (r_, re_, dth), items, frames, f, train_utts, test_utts); G[i, jj] = [rr[kk] for kk in keys]
            log(tag, 'r', r_, 'cv_utt', G[i, :, 0].round(4), 'cv_acc', G[i, :, 2].round(2), 'test_acc', G[i, :, 8].round(2), 'nmax %.2f' % G[i, :, 10].max()); np.savez(fp, **{f'grid_{tag}_partial': G}, **out)
        out[f'grid_{tag}'] = G; out['grid'] = grid; out['dth'] = dth; out['keys'] = np.array(keys); np.savez(fp, **out)
    log('done')
