"""Spoken digits with a winner-takes-all-aligned objective:
readout = ridge on frames with delta selected by 5-fold utterance-level cross-validation;
losses = utterance-level MSE of the averaged outputs (L_utt), softmax cross-entropy of the averaged outputs (L_CE),
both cross-validated over the training utterances; plus frame MSE (old), train/test WTA accuracy."""
from run_digits import *

def wta_task(dev, theta, items, frames, f, train_utts, test_utts, deltas=(0.1, 1, 10, 100, 1000), beta=5.0, folds=5, feats=None):
    if feats is None: feats, nmax = Reservoir(dev, theta).run(f)
    else: nmax = np.nan
    X = np.array([feats[idx].ravel() for (u, d, idx) in frames]); U = np.array([u for u, d, idx in frames]); D = np.array([d for u, d, idx in frames])
    tr = np.isin(U, train_utts); mu, sd = X[tr].mean(0), X[tr].std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    Zf = np.hstack([(X - mu) / sd, np.ones((len(X), 1))]); Y = np.eye(10)[D]
    def fit(rows, delta): return np.linalg.solve(Zf[rows].T @ Zf[rows] + delta * np.eye(Zf.shape[1]), Zf[rows].T @ Y[rows])
    def utt_outputs(W, utts):
        return np.array([(Zf[U == u] @ W).mean(0) for u in utts]), np.array([D[U == u][0] for u in utts])
    def losses(O, d):
        Yu = np.eye(10)[d]; l_utt = 0.5 * np.mean(np.sum((Yu - O) ** 2, 1))
        z = beta * O; z = z - z.max(1, keepdims=True); l_ce = -np.mean(z[np.arange(len(d)), d] - np.log(np.exp(z).sum(1)))
        return l_utt, l_ce, np.mean(np.argmax(O, 1) == d)
    rng = np.random.default_rng(0); tu = np.array(train_utts); perm = rng.permutation(len(tu)); fold_of = np.zeros(len(tu), int); fold_of[perm] = np.arange(len(tu)) % folds
    cv = {}
    for delta in deltas:
        O_all = np.zeros((len(tu), 10))
        for k in range(folds):
            hold = tu[fold_of == k]; keep = tu[fold_of != k]; W = fit(np.isin(U, keep), delta); O_all[fold_of == k] = utt_outputs(W, hold)[0]
        cv[delta] = losses(O_all, np.array([D[U == u][0] for u in tu]))
    dstar = min(deltas, key=lambda dl: cv[dl][0])
    W = fit(tr, dstar); Otr, dtr = utt_outputs(W, train_utts); Ote, dte = utt_outputs(W, test_utts)
    l_frame = 0.5 * np.mean((Y[tr] - Zf[tr] @ W) ** 2)
    return dict(cv_utt=cv[dstar][0], cv_ce=cv[dstar][1], cv_acc=cv[dstar][2], delta=dstar, train_utt=losses(Otr, dtr)[0], train_acc=losses(Otr, dtr)[2],
                test_utt=losses(Ote, dte)[0], test_ce=losses(Ote, dte)[1], test_acc=losses(Ote, dte)[2], frame_mse=l_frame, nmax=nmax,
                cv_utt_all=[cv[dl][0] for dl in deltas], cv_acc_all=[cv[dl][2] for dl in deltas])

if __name__ == '__main__':
    fp = os.path.join(DATA, 'digits_wta.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    items = load_subset(); f, frames = build_sequence(items)
    rng = np.random.default_rng(1); utts = np.arange(len(items)); digits = np.array([it[0] for it in items]); train_utts, test_utts = [], []
    for d in range(10):
        u = utts[digits == d]; rng.shuffle(u); train_utts += list(u[:len(u) // 2]); test_utts += list(u[len(u) // 2:])
    keys = ['cv_utt', 'cv_ce', 'cv_acc', 'delta', 'train_utt', 'train_acc', 'test_utt', 'test_ce', 'test_acc', 'frame_mse', 'nmax']
    if 'unsq' not in out:
        r = wta_task(DEV, (0, 0, 0), items, frames, f, train_utts, test_utts); out['unsq'] = np.array([r[k] for k in keys]); log('unsq', {k: r[k] for k in keys}, 'cv per delta', r['cv_utt_all'], r['cv_acc_all']); np.savez(fp, **out)
    reg, dg = np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 7)[:-1]
    if 'scan' not in out:
        S = np.zeros((6, 6, len(keys)))
        for j, re_ in enumerate(reg):
            for k, d in enumerate(dg):
                r = wta_task(DEV, (0.0, re_, d), items, frames, f, train_utts, test_utts); S[j, k] = [r[kk] for kk in keys]
            log('scan re', re_, 'cv_utt min/max %.4f %.4f' % (S[j, :, 0].min(), S[j, :, 0].max()), 'cv_acc', S[j, :, 2].round(2), 'test_acc', S[j, :, 8].round(2)); np.savez(fp, scan_partial=S, **{k_: v for k_, v in out.items()})
        out['scan'] = S; out['keys'] = np.array(keys); out['reg'] = reg; out['dg'] = dg; np.savez(fp, **out)
    if 'rline' not in out:
        R = np.array([[rr[k] for k in keys] for rr in (wta_task(DEV, (r_, 0.0, 0.0), items, frames, f, train_utts, test_utts) for r_ in reg)]); out['rline'] = R; log('rline cv_utt', R[:, 0].round(4), 'test_acc', R[:, 8].round(2)); np.savez(fp, **out)
    if 'flip' not in out:
        R = np.array([[rr[k] for k in keys] for rr in (wta_task(replace(DEV, s=-1.0), (r_, 0.0, 0.0), items, frames, f, train_utts, test_utts) for r_ in (0.3, 0.5))]); out['flip'] = R; log('flip cv_utt', R[:, 0].round(4), 'test_acc', R[:, 8].round(2)); np.savez(fp, **out)
    log('done')
