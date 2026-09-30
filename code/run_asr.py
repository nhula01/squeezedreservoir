"""Spoken-digit recognition with the protocol of github.com/nhula01/nonMarkovianReservoirComputer (SpeechRecognition):
5 speakers x 10 digits x 10 recordings (500 utterances); mel spectrogram (librosa, n_fft = len(audio), power_to_db ref=mean)
split into a 6x6 block grid; block means and block stds are two input channels of 36 values per utterance, each globally
normalized to [0,1]; utterances are fed consecutively (train block then test block, no separators); the reservoir is run
once per channel; features = (P, Q) [or all six observables] at the end of each of the 36 input steps -> 72 per channel,
144 per utterance; first 5 training utterances discarded as washout (395 train / 100 test); ten one-vs-all ridge
classifiers (delta = 1e-10, bias); winner-takes-all; word error rate; 5-fold cross-validation (KFold shuffle, seed 99)."""
from common import *
from scipy.io import wavfile
from librosa.feature import melspectrogram
from librosa import power_to_db
from sklearn.model_selection import KFold
import glob, re as regex

REC = os.path.join(os.environ.get('FSDD_DIR', os.path.expanduser('~/fsdd')), 'free-spoken-digit-dataset-master', 'recordings')   # set FSDD_DIR to the unzipped FSDD
NAMES = ['george', 'jackson', 'theo', 'lucas', 'nicolas']; NBLK = 6; NFT = NBLK * NBLK; REP = 10; WASH = 5

def load_audios():
    audios, y = [], []
    for name in NAMES:
        for d in range(10):
            files = sorted(glob.glob(f'{REC}/{d}_{name}_*.wav'), key=lambda f_: int(regex.findall(r'_(\d+)\.wav', f_)[0]))[:REP]
            for f_ in files:
                rate, data = wavfile.read(f_); audios.append(data.astype(float)); y.append(d)
    return audios, np.array(y), rate

def ft_mean_std(audios, n=NBLK):
    out = []
    for a in audios:
        sp = power_to_db(melspectrogram(y=a, n_fft=len(a)), ref=np.mean); row = []
        for v in np.array_split(sp, n, axis=0):
            for h in np.array_split(v, n, axis=1):
                if h.size == 0: row += [round(float(np.median(v)), 4), round(float(np.std(v)), 4)]
                else: row += [round(float(np.mean(h)), 4), round(float(np.std(h)), 4)]
        out.append(row)
    X = np.array(out).reshape(len(audios), n * n, 2)
    Xm, Xs = X[:, :, 0], X[:, :, 1]
    norm = lambda A: (A - min(A.min(), 0)) / (max(A.max(), 0) - min(A.min(), 0))     # as in the repository (min/max seeded at 0)
    return norm(Xm), norm(Xs)

def folds(n=500, seed=99):
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n)))

def run_channel(dev, theta, U, order, obs='PQ'):
    seq = U[order].ravel()                                  # inputs in [0,1] -> drive eps*u, as in the repository
    F, nmax = Reservoir(dev, theta).run(seq)
    Nv = dev.Nv; last = F[:, NOBS * (Nv - 1):NOBS * Nv]     # observables at the end of each symbol: Q, P, Q2, P2, sx, sy
    cols = [0, 1] if obs == 'PQ' else list(range(NOBS))
    return last[:, cols].reshape(len(order), NFT * len(cols)), nmax

def asr_fold(dev, theta, Xm, Xs, y, tr_idx, te_idx, obs='PQ', delta=1e-10):
    order = np.concatenate([tr_idx, te_idx]); n_tr = len(tr_idx)
    Fm, nm1 = run_channel(dev, theta, Xm, order, obs); Fs, nm2 = run_channel(dev, theta, Xs, order, obs)
    S = np.hstack([Fm, Fs]); Xtr, Xte = S[WASH:n_tr], S[n_tr:]; ytr, yte = y[order][WASH:n_tr], y[order][n_tr:]
    Xb = np.hstack([np.ones((len(Xtr), 1)), Xtr]); Xbt = np.hstack([np.ones((len(Xte), 1)), Xte])
    A = Xb.T @ Xb + delta * np.eye(Xb.shape[1]); Y = np.eye(10)[ytr]
    W = np.linalg.solve(A, Xb.T @ Y)
    L = 0.5 * np.mean((Y - Xb @ W) ** 2)
    pred = np.argmax(Xbt @ W, 1); wer = np.mean(pred != yte); wer_tr = np.mean(np.argmax(Xb @ W, 1) != ytr)
    return L, wer, wer_tr, max(nm1, nm2)

def frontend_fold(Xm, Xs, y, tr_idx, te_idx, delta=1e-10):
    S = np.hstack([Xm, Xs]); Xtr, Xte = S[tr_idx][WASH:], S[te_idx]; ytr, yte = y[tr_idx][WASH:], y[te_idx]
    Xb = np.hstack([np.ones((len(Xtr), 1)), Xtr]); Xbt = np.hstack([np.ones((len(Xte), 1)), Xte])
    W = np.linalg.solve(Xb.T @ Xb + delta * np.eye(Xb.shape[1]), Xb.T @ np.eye(10)[ytr])
    return np.mean(np.argmax(Xbt @ W, 1) != yte)

if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    fp = os.path.join(DATA, 'asr.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    fpf = os.path.join(DATA, 'asr_features.npz')
    if os.path.exists(fpf):
        d = np.load(fpf); Xm, Xs, y = d['Xm'], d['Xs'], d['y']
    else:
        audios, y, rate = load_audios(); Xm, Xs = ft_mean_std(audios); np.savez(fpf, Xm=Xm, Xs=Xs, y=y); log('features', Xm.shape, Xs.shape, np.bincount(y))
    FOLDS = folds()
    if 'frontend' not in out:
        out['frontend'] = np.array([frontend_fold(Xm, Xs, y, tr, te) for tr, te in FOLDS]); log('frontend ridge WER per fold', out['frontend'], out['frontend'].mean()); np.savez(fp, **out)
    if part in ('unsq', 'all') and 'unsq5' not in out:
        t0 = time.time(); r = [asr_fold(DEV, (0, 0, 0), Xm, Xs, y, tr, te) for tr, te in FOLDS]; out['unsq5'] = np.array(r); log('unsqueezed 5-fold (L, WER, WER_train, nmax)', np.array(r).round(4), 'mean WER %.3f' % np.mean([x[1] for x in r]), 'time', time.time() - t0)
        r6 = [asr_fold(DEV, (0, 0, 0), Xm, Xs, y, tr, te, obs='all') for tr, te in FOLDS]; out['unsq5_all'] = np.array(r6); log('unsqueezed 5-fold, six observables', np.array(r6).round(4), 'mean WER %.3f' % np.mean([x[1] for x in r6])); np.savez(fp, **out)
    tr0, te0 = FOLDS[0]
    if part in ('scan', 'all') and 'scan' not in out:
        reg, dg = np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 7)[:-1]; S = np.zeros((6, 6, 4))
        for j, re_ in enumerate(reg):
            for k, d_ in enumerate(dg): S[j, k] = asr_fold(DEV, (0.0, re_, d_), Xm, Xs, y, tr0, te0)
            log('scan fold0 re', re_, 'L', S[j, :, 0].round(5), 'WER', S[j, :, 1].round(2)); np.savez(fp, scan_partial=S, **out)
        out['scan'] = S; out['reg'] = reg; out['dg'] = dg; np.savez(fp, **out)
    if part in ('grid', 'all') and 'grid_m' not in out:
        S = out['scan']; j, k = np.unravel_index(S[:, :, 0].argmin(), (6, 6)); dth = float(out['dg'][k]); out['dth'] = dth; grid = np.linspace(0, 0.5, 6); out['grid'] = grid
        for s, tag in ((1.0, 'p'), (-1.0, 'm')):
            if f'grid_{tag}' in out: continue
            G = np.zeros((6, 6, 4))
            for i, r_ in enumerate(grid):
                for jj, re_ in enumerate(grid): G[i, jj] = asr_fold(replace(DEV, s=s), (r_, re_, dth), Xm, Xs, y, tr0, te0)
                log(tag, 'r', r_, 'L', G[i, :, 0].round(5), 'WER', G[i, :, 1].round(2), 'nmax %.2f' % G[i, :, 3].max()); out.pop(f'grid_{tag}_partial', None); np.savez(fp, **{f'grid_{tag}_partial': G}, **out)
            out[f'grid_{tag}'] = G; np.savez(fp, **out)
    if part in ('best5', 'all') and 'best5' not in out:
        cands = []
        for tag, s in (('p', 1.0), ('m', -1.0)):
            G = out[f'grid_{tag}']; i, jj = np.unravel_index(G[:, :, 0].argmin(), (6, 6)); cands.append((G[i, jj, 0], s, (out['grid'][i], out['grid'][jj], out['dth'])))
        S = out['scan']; j, k = np.unravel_index(S[:, :, 0].argmin(), (6, 6)); cands.append((S[j, k, 0], 1.0, (0.0, out['reg'][j], out['dg'][k])))
        Lb, s, th = min(cands, key=lambda c: c[0]); out['best'] = np.array(th); out['best_s'] = s
        r = [asr_fold(replace(DEV, s=s), th, Xm, Xs, y, tr, te) for tr, te in FOLDS]; out['best5'] = np.array(r); log('best', th, s, '5-fold', np.array(r).round(4), 'mean WER %.3f' % np.mean([x[1] for x in r]))
        r6 = [asr_fold(replace(DEV, s=s), th, Xm, Xs, y, tr, te, obs='all') for tr, te in FOLDS]; out['best5_all'] = np.array(r6); log('best 5-fold six obs', np.array(r6).round(4), 'mean WER %.3f' % np.mean([x[1] for x in r6])); np.savez(fp, **out)
    log('done', part)
