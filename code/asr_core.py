"""Spoken-digit protocol functions of run_asr.py without the audio frontend (features are read from data/asr_features.npz)."""

from common import *
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

