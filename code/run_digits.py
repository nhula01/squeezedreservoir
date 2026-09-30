"""Real-world task: spoken-digit recognition (Free Spoken Digit Dataset, 8 kHz).
Frontend: 8-channel log filterbank (32 ms frames, 16 ms hop), channels time-multiplexed into the single input.
Reservoir features: the 24 observables at each of the 8 channel sub-symbols of a frame (192 per frame).
Readout: ridge regression frame-by-frame onto one-hot digit targets; utterance class = argmax of the summed
outputs over its frames (Verstraeten/Appeltant protocol). Loss for optimization = training MSE of the frame readout."""
from common import *
from scipy.io import wavfile
import glob, re as regex

REC = os.path.join(os.environ.get('FSDD_DIR', os.path.expanduser('~/fsdd')), 'free-spoken-digit-dataset-master', 'recordings')   # set FSDD_DIR to the unzipped FSDD
NCH, NFFT, HOP = 8, 256, 128
SEP = 12                     # zero-input symbols between utterances (fading memory)

def filterbank(x, fs=8000):
    x = x.astype(float); x = x / (np.abs(x).max() + 1e-12)
    edges = np.geomspace(100, 3800, NCH + 2); fr = np.fft.rfftfreq(NFFT, 1 / fs)
    fb = np.zeros((NCH, len(fr)))
    for k in range(NCH):
        lo, c, hi = edges[k:k + 3]
        fb[k] = np.clip(np.minimum((fr - lo) / (c - lo), (hi - fr) / (hi - c)), 0, None)
    win = np.hanning(NFFT); frames = []
    for s in range(0, max(len(x) - NFFT, 1), HOP):
        seg = x[s:s + NFFT]; seg = np.pad(seg, (0, NFFT - len(seg)))
        P = np.abs(np.fft.rfft(seg * win)) ** 2; frames.append(np.log(fb @ P + 1e-8))
    return np.array(frames)                        # (T, NCH)

def load_subset(speakers=('jackson', 'nicolas'), per_digit=6, seed=0):
    rng = np.random.default_rng(seed); items = []
    for sp in speakers:
        for d in range(10):
            files = sorted(glob.glob(f'{REC}/{d}_{sp}_*.wav'), key=lambda f: int(regex.findall(r'_(\d+)\.wav', f)[0]))
            idx = rng.choice(len(files), per_digit, replace=False)
            for i in idx:
                fs, x = wavfile.read(files[i]); items.append((d, sp, filterbank(x, fs)))
    return items

def build_sequence(items):
    """Concatenate utterances into one encoded input sequence; return f (T,), frame index map."""
    allf = np.concatenate([it[2] for it in items]); lo, hi = np.percentile(allf, 1), np.percentile(allf, 99)
    seq, frames = [], []          # frames: list of (utterance id, digit, [symbol indices of its 8 sub-symbols])
    for u, (d, sp, F) in enumerate(items):
        seq += [0.0] * SEP
        for fr in F:
            start = len(seq); seq += list(np.clip(2 * (fr - lo) / (hi - lo) - 1, -1, 1)); frames.append((u, d, list(range(start, start + NCH))))
    return np.array(seq), frames

def digit_task(dev, theta, items, frames, f, train_utts, test_utts, delta=None, feats=None, return_feats=False):
    if feats is None:
        feats, nmax = Reservoir(dev, theta).run(f)
    else:
        nmax = np.nan
    X = np.array([feats[idx].ravel() for (u, d, idx) in frames]); U = np.array([u for u, d, idx in frames]); D = np.array([d for u, d, idx in frames])
    tr = np.isin(U, train_utts); te = np.isin(U, test_utts)
    mu, sd = X[tr].mean(0), X[tr].std(0); sd = np.where(sd > 1e-12, sd, 1.0); Z = (X - mu) / sd
    Zf = np.hstack([Z, np.ones((len(Z), 1))]); Y = np.eye(10)[D]
    if delta is None: delta = tr.sum() * dev.sigma_rel ** 2
    W = np.linalg.solve(Zf[tr].T @ Zf[tr] + delta * np.eye(Zf.shape[1]), Zf[tr].T @ Y[tr])
    out = Zf @ W; L = 0.5 * np.mean((Y[tr] - out[tr]) ** 2)
    def acc(mask, utts):
        ok = 0
        for u in utts:
            m = (U == u) & mask; ok += int(np.argmax(out[m].sum(0)) == D[m][0])
        return ok / len(utts)
    r_ = (L, acc(tr, train_utts), acc(te, test_utts), nmax)
    return (r_, feats) if return_feats else r_

def frontend_baseline(items, frames, f, train_utts, test_utts, delays=1):
    """Ridge on the filterbank frames themselves (optionally with delayed frames), same protocol."""
    F = np.array([f[idx] for (u, d, idx) in frames]); U = np.array([u for u, d, idx in frames]); D = np.array([d for u, d, idx in frames])
    if delays > 1:
        F = np.hstack([np.vstack([np.zeros((k, NCH)), F[:len(F) - k]]) for k in range(delays)])
    tr = np.isin(U, train_utts); te = np.isin(U, test_utts)
    Zf = np.hstack([F, np.ones((len(F), 1))]); Y = np.eye(10)[D]
    W = np.linalg.solve(Zf[tr].T @ Zf[tr] + 1e-3 * np.eye(Zf.shape[1]), Zf[tr].T @ Y[tr]); out = Zf @ W
    def acc(mask, utts):
        return np.mean([np.argmax(out[(U == u) & mask].sum(0)) == D[U == u][0] for u in utts])
    return acc(tr, train_utts), acc(te, test_utts)

if __name__ == '__main__':
    fp = os.path.join(DATA, 'digits.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    items = load_subset(); f, frames = build_sequence(items)
    rng = np.random.default_rng(1); utts = np.arange(len(items)); digits = np.array([it[0] for it in items])
    train_utts, test_utts = [], []
    for d in range(10):                               # stratified 50/50 split
        u = utts[digits == d]; rng.shuffle(u); train_utts += list(u[:len(u) // 2]); test_utts += list(u[len(u) // 2:])
    log('sequence length', len(f), 'utterances', len(items), 'frames', len(frames))
    if 'base' not in out:
        out['base'] = np.array([frontend_baseline(items, frames, f, train_utts, test_utts, k) for k in (1, 3, 10)]); log('frontend baselines (train acc, test acc) for 1,3,10 frames', out['base'])
    if 'L0' not in out:
        t0 = time.time(); r0 = digit_task(DEV, (0, 0, 0), items, frames, f, train_utts, test_utts); out['L0'] = np.array(r0); log('unsqueezed', r0, 'time', time.time() - t0); np.savez(fp, **out)
    if 'scan' not in out:
        reg, dg = np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 7)[:-1]; S = np.zeros((6, 6, 4))
        for j, re_ in enumerate(reg):
            for k, d in enumerate(dg):
                S[j, k] = digit_task(DEV, (0.0, re_, d), items, frames, f, train_utts, test_utts)
            log('scan re', re_, 'min L', S[j, :, 0].min(), 'best test acc', S[j, :, 2].max()); np.savez(fp, scan_partial=S, **{k_: v for k_, v in out.items()})
        out['scan'] = S; out['scan_re'] = reg; out['scan_dth'] = dg; np.savez(fp, **out)
    if 'rline' not in out:
        rg = np.linspace(0, 0.5, 6); R = np.array([digit_task(DEV, (r, 0.0, 0.0), items, frames, f, train_utts, test_utts) for r in rg])
        out['rline'] = R; out['rline_r'] = rg; log('r line', R[:, [0, 2]]); np.savez(fp, **out)
    if 'gd' not in out:
        S = out['scan']; j, k = np.unravel_index(S[:, :, 0].argmin(), S[:, :, 0].shape)
        fun = lambda th: digit_task(DEV, project(th), items, frames, f, train_utts, test_utts)[0]
        class C:
            def __init__(s): s.n = 0; s.trace = []
            def __call__(s, th):
                if s.n >= 40: raise StopIteration
                L = fun(th); s.n += 1; s.trace.append((s.n, *project(th), L)); return L
        res = []
        for th0 in (np.array([0.25, 0.25, 1.0]), np.array([0.1, 0.4, -2.0])):
            c = C(); gradient_descent(c, th0, 40, h=np.array([3e-2, 3e-2, 3e-2])); tr_ = np.array(c.trace); ib = tr_[:, 4].argmin()
            full = digit_task(DEV, tr_[ib, 1:4], items, frames, f, train_utts, test_utts); res.append([*tr_[ib, 1:5], full[1], full[2]]); log('gd', tr_[0, 4], '->', tr_[ib, 1:5], full)
            np.savez(fp, gd_partial=np.array(res), **{k_: v for k_, v in out.items()})
        out['gd'] = np.array(res); np.savez(fp, **out)
    log('done digits')
