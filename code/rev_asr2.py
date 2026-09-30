"""Spoken digits against a tuned drive. Protocol of run_asr.py (features from data/asr_features.npz), operating points
selected by the first-fold training loss only. (1) drive amplitude and frequency at zero squeezing: eps in {0.1, 0.2, 0.4}
x delta in {-0.5, -0.25, 0, 0.25, 0.5}; (2) squeezing from that base: (r, r_e) in {0, 0.25, 0.5}^2, dth in {0, pi},
phi_d in {0, pi/2}; (3) five-fold WER at the unsqueezed fabricated device, the tuned base and the squeezed tuned device;
cutoff check at the selected point."""
from asr_core import *
if __name__ == '__main__':
    fp = os.path.join(DATA, 'rev_asr2.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    d = np.load(os.path.join(DATA, 'asr_features.npz')); Xm, Xs, y = d['Xm'], d['Xs'], d['y']; F = folds(); tr0, te0 = F[0]
    nc = lambda e: 8 if e <= 0.3 else 10
    if 'base' not in out:
        rows = []
        for e in (0.1, 0.2, 0.4):
            for dl in (-0.5, -0.25, 0.0, 0.25, 0.5):
                dv = replace(DEV, eps=e, omega=1 + dl, omega_q=1 + dl, Nc=nc(e)); rows.append([e, dl, *asr_fold(dv, (0, 0, 0), Xm, Xs, y, tr0, te0)])
            log('base eps', e, np.array(rows)[-5:, 2].round(5))
        out['base'] = np.array(rows); np.savez(fp, **out)
    B = out['base']; ok = B[:, 5] <= 2.5; ib = np.where(ok)[0][B[ok, 2].argmin()]; e2, dl2 = B[ib, :2]
    base = replace(DEV, eps=e2, omega=1 + dl2, omega_q=1 + dl2, Nc=nc(e2)); log('tuned base', e2, dl2, B[ib, 2:])
    if 'scan' not in out:
        rows = []
        for pd in (0.0, np.pi / 2):
            for r in (0, 0.25, 0.5):
                for re in (0, 0.25, 0.5):
                    for dth in (0.0, np.pi):
                        if r == 0 and re == 0 and dth != 0: continue
                        rows.append([pd, r, re, dth, *asr_fold(replace(base, phi_d=pd), (r, re, dth), Xm, Xs, y, tr0, te0)])
            log('scan phi_d', pd, 'best L', np.array(rows)[:, 4].min()); out['scan'] = np.array(rows); np.savez(fp, **out)
    S = out['scan']; ok = S[:, 7] <= 2.5; js = np.where(ok)[0][S[ok, 4].argmin()]; pd, th = S[js, 0], S[js, 1:4]
    if 'wer5' not in out:
        a = [asr_fold(DEV, (0, 0, 0), Xm, Xs, y, tr, te) for tr, te in F]
        b = [asr_fold(base, (0, 0, 0), Xm, Xs, y, tr, te) for tr, te in F]
        c = [asr_fold(replace(base, phi_d=pd), th, Xm, Xs, y, tr, te) for tr, te in F]
        conv = asr_fold(replace(base, phi_d=pd, Nc=base.Nc + 4), th, Xm, Xs, y, tr0, te0)
        out['wer5'] = np.array([a, b, c]); out['sel'] = np.array([e2, dl2, pd, *th]); out['conv'] = np.array(conv)
        log('WER5 fabricated %.3f tuned %.3f squeezed %.3f' % tuple(np.array([a, b, c])[:, :, 1].mean(1)), 'sel', out['sel'].round(3), 'conv', conv, 'first-fold L', b[0][0], c[0][0])
        np.savez(fp, **out)
