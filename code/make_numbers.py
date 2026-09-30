import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data'); PAPER = os.path.join(HERE, '..', 'paper'); os.makedirs(PAPER, exist_ok=True)
def ld(n):
    p = os.path.join(DATA, n); return np.load(p, allow_pickle=True) if os.path.exists(p) else None
M = {}
def m(k, v): M[k] = v
def sci(x, d=2):
    e = int(np.floor(np.log10(abs(x)))); return r'\ensuremath{%.*f\times10^{%d}}' % (d, x / 10 ** e, e)
m('rmax', '0.5'); m('sigmarel', '0.01'); m('ridgeDelta', '0.07'); m('truncTol', '1'); m('nInit', '16'); m('budgetMG', '100')
sc = ld('fig2_scans.npz')
Lstar_c = []
if sc is not None:
    L3 = sc['L3']; L0 = float(L3[0, 0, 0]); NR0 = float(sc['N3'][0, 0, 0])
    m('Lzero', sci(L0)); m('NRMSEzero', '%.3f' % NR0)
    i, j, k = np.unravel_index(L3.argmin(), L3.shape)
    m('bestCr', '%.1f' % sc['r3'][i]); m('bestCre', '%.1f' % sc['re3'][j]); m('bestCd', '%.2f' % sc['d3'][k]); m('LbestCoarse', sci(L3.min()))
    m('reFix', '%.2f' % sc['re_fix']); m('dthFix', '%.2f' % sc['dth_fix'])
    L1, L2 = sc['L1'], sc['L2']
    if L1.min() <= L2.min():
        a, b = np.unravel_index(L1.argmin(), L1.shape); best = (sc['rg'][a], sc['re_fix'], sc['dg'][b]); Lb = L1.min(); NRb = sc['NR1'][a, b]; nmb = sc['NM1'][a, b]
    else:
        a, b = np.unravel_index(L2.argmin(), L2.shape); best = (sc['rg'][a], sc['reg'][b], sc['dth_fix']); Lb = L2.min(); NRb = sc['NR2'][a, b]; nmb = sc['NM2'][a, b]
    m('bestSr', '%.2f' % best[0]); m('bestSre', '%.2f' % best[1]); m('bestSd', '%.2f' % best[2]); m('LbestSlice', sci(Lb))
    m('imprBest', '%.0f' % (100 * (1 - Lb / L0))); m('NRMSEbest', '%.3f' % NRb); m('nmaxBest', '%.2f' % nmb)
    m('LrangeFactor', '%.1f' % (max(L1.max(), L2.max()) / min(L1.min(), L2.min())))
    if 'T1' in sc:
        bad = (sc['T1'] > 0.01) | (sc['T2'] > 0.01)
        # photon number in bad cells: use NM slices at nearest grid points
        nm_bad = []
        for ii in range(7):
            for jj in range(7):
                if sc['T1'][ii, jj] > 0.01:
                    ia = np.argmin(abs(sc['rg'] - sc['rc'][ii])); ib = np.argmin(abs(sc['dg'] - sc['dc'][jj])); nm_bad.append(sc['NM1'][ia, ib])
                if sc['T2'][ii, jj] > 0.01:
                    ia = np.argmin(abs(sc['rg'] - sc['rc'][ii])); ib = np.argmin(abs(sc['reg'] - sc['rec'][jj])); nm_bad.append(sc['NM2'][ia, ib])
        m('nmaxTrunc', '%.1f' % (min(nm_bad) if nm_bad else 0))
    Lr = L2[:, 0]; m('imprRonly', '%.0f' % (100 * (1 - Lr.min() / L0))); m('rOnlyBest', '%.2f' % sc['rg'][Lr.argmin()])
    Lstar_c += [L3.min(), L1.min(), L2.min()]
f3 = ld('fig3_mg.npz')
if f3 is not None and sc is not None:
    B = int(f3['budget']); m('budgetMG', str(B)); m('nInit', str(len(f3['inits'])))
    Lstar_c += [f3[f'final_{k}'][:, 3].min() for k in ('gd', 'nm', 'rs')]
    Lstar = min(Lstar_c); m('Lstar', sci(Lstar))
    for k in ('gd', 'nm', 'rs'):
        C = f3[f'curve_{k}']; fin = C[:, -1]; ini = C[:, 0]
        m(k + 'RedMedian', '%.0f' % (100 * np.median(1 - fin / ini)))
        m(k + 'FracBetter', '%d' % np.sum(fin < L0))
        q1, q3 = np.percentile(fin, [25, 75]); m(k + 'IQR', sci(q3 - q1, 1))
        reach = [(np.argmax(C[r] <= 1.05 * Lstar) + 1) if np.any(C[r] <= 1.05 * Lstar) else np.inf for r in range(len(C))]
        med = np.median(reach); m(k + 'EvalsFive', ('%.0f' % med) if np.isfinite(med) else 'more than %d' % B)
        m(k + 'FracReach', '%d' % np.sum(np.isfinite(reach)))
        for tol, nm_ in ((0.02, 'Two'), (0.01, 'One')):
            reach = [(np.argmax(C[r] <= (1 + tol) * Lstar) + 1) if np.any(C[r] <= (1 + tol) * Lstar) else np.inf for r in range(len(C))]
            m(k + 'Reach' + nm_, '%d' % np.sum(np.isfinite(reach)))
            fin_r = [x for x in reach if np.isfinite(x)]
            m(k + 'Evals' + nm_, '%.0f' % np.mean(fin_r) if fin_r else '--')
        m(k + 'FinalMedian', sci(np.median(fin))); m(k + 'FinalMax', sci(fin.max()))
    fg = f3['final_gd']; m('gdSecondary', '%d' % np.sum((fg[:, 3] > 1.01 * Lstar) & (fg[:, 3] < 1.05 * Lstar))); m('gdTrapped', '%d' % np.sum(fg[:, 3] >= 1.05 * Lstar))
    m('basinFrac', '%.0f' % (100 * np.mean(sc['L3'] <= 1.05 * Lstar)))
    fg = f3['final_gd']; m('gdFracRzero', '%d of %d' % (np.sum(fg[:, 0] < 0.02), len(fg)))
f4 = ld('fig4_lines.npz')
if f4 is not None:
    Sg = f4['line_r'][0, 12:21].reshape(3, 3); fr = Sg ** 2 / (Sg ** 2).sum(1, keepdims=True)
    for i, nm_ in enumerate(('R', 'Re', 'Dth')):
        m('fracMean' + nm_, '%.0f' % (100 * fr[i, 0]) if fr[i, 0] >= 0.005 else '%.1f' % (100 * fr[i, 0])); m('fracSec' + nm_, '%.0f' % (100 * fr[i, 1])); m('fracEm' + nm_, '%.0f' % (100 * fr[i, 2]))
    F = f4['init_r']; ok = F[:, 4] <= 1.01 * min(Lstar_c); m('rTrapBelow', '%.2f' % F[ok, 0].max()); m('rTrapAbove', '%.2f' % F[~ok, 0].min() if (~ok).any() else '--')
    A = f4['line_re']; m('qfiRe', '%.0f' % np.median(A[:, 7]))
    A = f4['line_dth']; m('qfiDth', sci(np.median(A[:, 8]), 1))
    fins = np.concatenate([f4['init_r'][:, 4], f4['init_re'][:, 4]]) if 'init_re' in f4 else f4['init_r'][:, 4]
    ref = min(Lstar_c) if Lstar_c else fins.min()
    m('initSpread', '%.0f' % (100 * (np.max(fins) / ref - 1)))
    m('initSpreadMedian', '%.0f' % (100 * (np.median(fins) / ref - 1)))
fd = ld('supp_fd.npz')
if fd is not None:
    hs = fd['fd_h']
    def spread(G, hmax, comps):
        ref = G[hs == 1e-3][0]; sel = hs <= hmax
        return np.max(np.abs(G[sel][:, comps] - ref[comps]) / np.abs(ref[comps]))
    m('fdSpread', '%.1f' % (100 * max(spread(fd['fd_unsqueezed'], 1e-2, [0, 1, 2]), spread(fd['fd_mid'], 1e-2, [0, 1, 2]))))
    m('fdSpreadWide', '%.0f' % (100 * max(spread(fd['fd_unsqueezed'], 3e-2, [0, 1, 2]), spread(fd['fd_mid'], 3e-2, [0, 1, 2]))))
    m('fdSpreadBasinR', '%.1f' % (100 * spread(fd['fd_basin'], 1e-2, [0])))
    m('fdSpreadBasinRe', '%.0f' % (100 * spread(fd['fd_basin'], 1e-2, [1])))
    m('fdSpreadBasinReSmall', '%.0f' % (100 * spread(fd['fd_basin'], 3e-3, [1])))
    G = fd['fd_mid'][hs == 1e-2][0]; m('gradMid', '(%s, %s, %s)' % tuple(sci(x, 1) for x in G))
    G = fd['fd_basin'][hs == 1e-2][0]; m('gradBasin', '(%s, %s, %s)' % tuple(sci(x, 1) for x in G))
nz = ld('supp_noise.npz'); ng = ld('supp_noisygd.npz')
if nz is not None:
    hs, sig = nz['hs'], nz['sig']
    for name in ('mid', 'basin'):
        G0, Gn = nz[f'G0_{name}'], nz[f'Gn_{name}']
        a = np.argmin(abs(hs - 1e-1)); b = np.argmin(abs(sig - 1e-2))
        fr = [np.mean(np.sign(Gn[a, b, :, i]) == np.sign(G0[a, i])) for i in range(3)]
        m('signR' + name.capitalize(), '%.0f' % (100 * fr[0])); m('signRe' + name.capitalize(), '%.0f' % (100 * fr[1])); m('signDth' + name.capitalize(), '%.0f' % (100 * fr[2]))
        a3 = np.argmin(abs(hs - 3e-2)); fr3 = [np.mean(np.sign(Gn[a3, b, :, i]) == np.sign(G0[a3, i])) for i in range(3)]
        m('signRhSmall' + name.capitalize(), '%.0f' % (100 * fr3[0])); m('signRehSmall' + name.capitalize(), '%.0f' % (100 * fr3[1]))
        dmax = max(np.mean(np.sign(Gn[aa, bb, :, 2]) == np.sign(G0[aa, 2])) for aa in range(3) for bb in range(4))
        m('signDthMax' + name.capitalize(), '%.0f' % (100 * dmax))
if ng is not None and 'clean_0.01' in ng:
    m('noisyGDRed', '%.0f' % (100 * np.median(1 - ng['clean_0.01'] / ng['L0_clean'])))
    m('noisyGDRedLow', '%.0f' % (100 * np.median(1 - ng['clean_0.003'] / ng['L0_clean'])))
    m('noisyGDMax', sci(ng['clean_0.01'].max())); m('noisyGDMaxLow', sci(ng['clean_0.003'].max()))
ns = ld('supp_narma_scan.npz')
if ns is not None:
    L3n = ns['L3']; m('imprNarma', '%.0f' % (100 * (1 - L3n.min() / L3n[0, 0, 0]))); m('LzeroNarma', sci(L3n[0, 0, 0])); m('LbestNarma', sci(L3n.min()))
    i, j, k = np.unravel_index(L3n.argmin(), L3n.shape); m('bestNarma', '(%.1f, %.1f, %.2f)' % (ns['r3'][i], ns['re3'][j], ns['d3'][k]))
fn = ld('fig3_narma.npz')
if fn is not None:
    C = fn['curve_gd']; m('narmaGDFrac', '%d of %d' % (np.sum(C[:, -1] < C[:, 0]), len(C)))
    m('narmaGDRed', '%.0f' % (100 * np.median(1 - C[:, -1] / C[:, 0])))
fk = ld('supp_fock.npz')
if fk is not None:
    key = 'fock_best_gd' if 'fock_best_gd' in fk else 'fock_best_coarse'
    A = fk[key]; m('fockConv', sci(abs(A[1, 1] - A[-1, 1]) / A[-1, 1], 1))
rd = ld('supp_ridge.npz')
if rd is not None:
    from scipy.stats import spearmanr
    L = rd['L']; sigs = rd['sigs']; ref = np.argmin(abs(sigs - 0.01))
    for a, s in enumerate(sigs):
        tag = {0.003: 'Lo', 0.01: 'Ref', 0.03: 'Hi', 0.1: 'VHi'}[float(s)]
        m('ridgeRho' + tag, '%.2f' % spearmanr(L[ref].ravel(), L[a].ravel()).correlation)
        i, j, k = np.unravel_index(L[a].argmin(), L[a].shape)
        m('ridgeMin' + tag, '(%.1f, %.1f, %.2f)' % (rd['r3'][i], rd['re3'][j], rd['d3'][k]))
        m('ridgeImpr' + tag, '%.0f' % (100 * (1 - L[a].min() / L[a][0, 0, 0])))
TASKN_ = {'mg': 'Mackey--Glass', 'narma': 'NARMA10', 'lorenz': 'Lorenz', 'nce': 'channel equalization'}

tk = ld('tasks.npz')
if tk is not None:
    TASKU = {'mg': 'MG', 'narma': 'NARMA', 'lorenz': 'LORENZ', 'nce': 'NCE'}
    recov = {}
    for nm, U in TASKU.items():
        if f'{nm}_best' not in tk: continue
        L3 = tk[f'{nm}_L3']; L0 = L3[0, 0, 0]; Lb = float(tk[f'{nm}_bestL']); Lr = tk[f'{nm}_refab'].min()
        m('impr' + U, '%.0f' % (100 * (1 - Lb / L0))); m('imprRefab' + U, '%.0f' % (100 * (1 - Lr / L0)))
        rec = (L0 - Lb) / (L0 - Lr) if L0 > Lr else np.inf
        recov[nm] = rec; m('recov' + U, '%.0f' % (100 * rec) if np.isfinite(rec) else '--')
        N3 = tk[f'{nm}_N3']; gd = tk[f'{nm}_gd']
        nrb = gd[gd[:, 3].argmin(), 4] if (len(gd) and gd[:, 3].min() <= L3.min()) else N3.ravel()[L3.argmin()]
        m('nrmse' + U + 'zero', '%.3f' % N3[0, 0, 0]); m('nrmse' + U + 'best', '%.3f' % nrb)
        b = tk[f'{nm}_best']; m('best' + U, '(%.2f, %.2f, %.2f)' % tuple(b))
        e0 = dict(tk[f'{nm}_eff0']); e1 = dict(tk[f'{nm}_eff1'])
        gch = max(abs(float(e1[k]) / float(e0[k]) - 1) for k in ('gamma_x', 'gamma_y', 'gamma_z'))
        m('gammaChange' + U, '%.0f' % (100 * gch))
        m('epsEff' + U, '%.3f' % float(e1['eps_eff'])); m('gco' + U, '%.3f' % float(e1['g_co'])); m('gcr' + U, '%.3f' % float(e1['g_cr'])); m('Omega' + U, '%.3f' % float(e1['Omega']))
        C = tk[f'{nm}_gdcurves']; m('gdWithin' + U, '%.1f' % (100 * (C[:, -1].max() / Lb - 1))); m('gdGood' + U, '%d of %d' % (np.sum(C[:, -1] <= 1.05 * Lb), len(C))); m('gdWorstImpr' + U, '%.0f' % (100 * (1 - C[:, -1].max() / L0)))
        i, j, k = np.unravel_index(tk[f'{nm}_refab'].argmin(), tk[f'{nm}_refab'].shape)
        m('refabBest' + U, '(%.2f, %.1f, %.2f)' % (np.linspace(0.25, 0.9, 6)[i], np.linspace(0.1, 0.6, 6)[j], np.linspace(0.1, 0.3, 5)[k]))
    fin = [v for v in recov.values() if np.isfinite(v) and v < 1]
    m('recovMin', '%.0f\\%%' % (100 * min(fin)) if fin else '--'); m('recovMax', '%.0f\\%%' % (100 * max(fin)) if fin else '--')
    beat = [TASKN_[k] for k, v in recov.items() if v >= 1]
    m('nceStatement', ', '.join(beat) if beat else 'none of the tasks')
    if 'nce' in recov:
        L0 = tk['nce_L3'][0, 0, 0]; Lb = float(tk['nce_bestL']); Lr = tk['nce_refab'].min(); m('nceVsRefab', '%.0f\\%%' % (100 * (1 - Lb / Lr)))
    m('refabNmax', '%.2f' % max(tk[f'{nm}_refabNM'].max() for nm in TASKU if f'{nm}_refabNM' in tk))
    if 'effkeys' in tk:
        keys = list(tk['effkeys']); ix = {k: i for i, k in enumerate(keys)}
        A = tk['effline_re']; g0 = A[0, [ix['gamma_x'], ix['gamma_y'], ix['gamma_z']]]
        m('gammaRangeRe', '%.0f' % (100 * np.max(np.abs(A[:, [ix['gamma_x'], ix['gamma_y'], ix['gamma_z']]] / g0 - 1))))
        A = tk['effline_dth']; gm = A[:, [ix['gamma_x'], ix['gamma_y'], ix['gamma_z']]]
        m('gammaRangeDth', '%.1f' % (100 * np.max((gm.max(0) - gm.min(0)) / gm.mean(0))))
    pass

lf = ld('lorenz_flip.npz'); nc = ld('nce_conv.npz')
if lf is not None and tk is not None:
    L0 = tk['lorenz_L3'][0, 0, 0]; Lb = lf['L3'].min(); Lr = tk['lorenz_refab'].min()
    m('imprLORENZflip', '%.0f' % (100 * (1 - Lb / L0))); m('recovLORENZflip', '%.0f' % (100 * (L0 - Lb) / (L0 - Lr)))
    m('bestLORENZflip', '(%.2f, %.2f, %.2f)' % tuple(lf['best'])); e1 = dict(lf['eff1']); m('epsEffLORENZflip', '%.3f' % float(e1['eps_eff']))
    m('nrmseLORENZflip', '%.3f' % lf['N3'].ravel()[lf['L3'].argmin()]); m('convLORENZflip', '%.1f' % (100 * abs(lf['conv'][0, 0] - lf['conv'][2, 0]) / lf['conv'][2, 0]))
if nc is not None:
    m('nceLconv', sci(nc['L'][2])); m('nceConvRel', '%.0f' % (100 * abs(nc['L'][0] - nc['L'][2]) / nc['L'][2])); m('nrmseNCEconv', '%.3f' % nc['NR'][2])
    if tk is not None: m('imprNCEconv', '%.0f' % (100 * (1 - nc['L'][2] / tk['nce_L3'][0, 0, 0])))
R3_, RE3_, D3_ = np.linspace(0, 0.5, 6), np.linspace(0, 0.5, 6), np.linspace(-np.pi, np.pi, 13)[:-1]

rv = ld('review.npz')
if rv is not None and tk is not None:
    TASKU = {'mg': 'MG', 'narma': 'NARMA', 'lorenz': 'LORENZ', 'nce': 'NCE'}
    if 'gap_re' in rv:
        m('gapZero', '%.3f' % rv['gap_re'][0]); m('gapReMax', '%.3f' % rv['gap_re'][-1]); m('gapReRise', '%.0f' % (100 * (rv['gap_re'][-1] / rv['gap_re'][0] - 1)))
        m('gapDthRange', '%.1f' % (100 * np.ptp(rv['gap_dth']) / rv['gap_dth'].mean())); m('gapRMax', '%.3f' % rv['gap_r'][-1])
    for nm, U in TASKU.items():
        if f'base_{nm}' in rv:
            for row in rv[f'base_{nm}']:
                m('base' + U + {'ols4': 'OLSfour', 'ols10': 'OLSten', 'quad10': 'Quadten', 'esn100': 'ESN'}[row[0]], '%.3f' % float(row[2]))
        if f'refabx_{nm}' in rv:
            L0 = tk[f'{nm}_L3'][0, 0, 0]; Lb = float(tk[f'{nm}_bestL']); Lr2 = min(tk[f'{nm}_refab'].min(), rv[f'refabx_{nm}'].min())
            m('imprRefabX' + U, '%.0f' % (100 * (1 - Lr2 / L0))); rec = (L0 - Lb) / (L0 - Lr2) if L0 > Lr2 else np.inf
            m('recovX' + U, '%.0f' % (100 * rec) if np.isfinite(rec) else '--'); m('refabxNmax' + U, '%.2f' % rv[f'refabxNM_{nm}'].max())
        if f'seeds_{nm}' in rv:
            S = rv[f'seeds_{nm}']; red = 100 * (1 - S[:, 1] / S[:, 0])
            m('seeds' + U, '%.0f and %.0f' % tuple(red)); m('seedsNR' + U, '%.3f/%.3f and %.3f/%.3f' % (S[0, 2], S[0, 3], S[1, 2], S[1, 3]))
    if 'seeds_lorenzflip' in rv:
        S = rv['seeds_lorenzflip']; m('seedsLORENZflip', '%.0f and %.0f' % tuple(100 * (1 - S[:, 1] / S[:, 0])))
    if 'refabx_lorenz' in rv and lf is not None:
        L0 = tk['lorenz_L3'][0, 0, 0]; Lb = lf['L3'].min(); Lr2 = min(tk['lorenz_refab'].min(), rv['refabx_lorenz'].min()); m('recovXLORENZflip', '%.0f' % (100 * (L0 - Lb) / (L0 - Lr2)))
    for nm, U in (('nce', 'NCE'), ('lorenzflip', 'LORENZflip')):
        if f'box_{nm}' in rv:
            L0 = tk['nce_L3'][0, 0, 0] if nm == 'nce' else tk['lorenz_L3'][0, 0, 0]
            b = rv[f'boxbest_{nm}']; c = rv[f'boxconv_{nm}']
            m('boxBest' + U, '(%.1f, %.1f, %.2f)' % tuple(b)); m('boxImpr' + U, '%.0f' % (100 * (1 - c[1, 0] / L0))); m('boxConv' + U, '%.1f' % (100 * abs(c[0, 0] - c[1, 0]) / c[1, 0]))
            m('boxNR' + U, '%.3f' % c[1, 1]); m('boxNmax' + U, '%.1f' % c[1, 2]); m('boxEdge' + U, 'yes' if max(b[0], b[1]) >= 0.79 else 'no')
    if 'dev2_L3' in rv:
        L3 = rv['dev2_L3']; L0 = L3[0, 0, 0]; i, j, k = np.unravel_index(L3.argmin(), L3.shape)
        m('devTwoLzero', sci(L0)); m('devTwoNRzero', '%.3f' % rv['dev2_N3'][0, 0, 0]); m('devTwoImpr', '%.0f' % (100 * (1 - L3.min() / L0))); m('devTwoBest', '(%.1f, %.1f, %.2f)' % (R3_[i], RE3_[j], D3_[k]))
        gd = rv['dev2_gd']; m('devTwoGDImpr', '%.0f' % max(0.0, 100 * (1 - gd[:, 3].min() / L0))); m('devTwoGDr', ', '.join('%.2f' % x for x in gd[:, 0]))

ls_ = ld('laser.npz'); dg = ld('digits.npz')
if ls_ is not None and 'best' in ls_:
    L0 = ls_['L3'][0, 0, 0]; Lb = float(ls_['bestL']); NR0 = ls_['N3'][0, 0, 0]
    m('laserImpr', '%.0f' % (100 * (1 - Lb / L0))); m('laserNRzero', '%.3f' % NR0); m('laserNRbest', '%.3f' % ls_['conv'][0, 1]); m('laserNRbestConv', '%.3f' % ls_['conv'][2, 1])
    b = ls_['best']; m('laserBest', '(%.2f, %.2f, %.2f)' % tuple(b)); m('laserBestDth', '%.2f' % b[2])
    for row in ls_['base']: m('laserBase' + {'ols4': 'OLSfour', 'ols10': 'OLSten', 'quad10': 'Quadten', 'esn100': 'ESN'}[row[0]], '%.3f' % float(row[2]))
    C = ls_['gdcurves']; m('laserGDGood', '%d' % np.sum(C[:, -1] <= 1.05 * Lb)); m('laserGDRange', '%.0f--%.0f' % (100 * (1 - C[:, -1].max() / L0), 100 * (1 - C[:, -1].min() / L0)))
    if 'refab' in ls_:
        Lr = ls_['refab'].min(); m('laserRefabImpr', '%.0f' % (100 * (1 - Lr / L0))); rec = (L0 - Lb) / (L0 - Lr) if L0 > Lr else np.inf; m('laserRecov', '%.0f' % (100 * rec) if np.isfinite(rec) else '--'); m('laserRefabNmax', '%.2f' % ls_['refabNM'].max())
    m('laserConv', '%.1f' % (100 * abs(ls_['conv'][0, 0] - ls_['conv'][2, 0]) / ls_['conv'][2, 0])); m('laserNmax', '%.2f' % ls_['conv'][2, 2])
    S = ls_['seeds']; m('laserSeeds', '%.0f and %.0f' % tuple(100 * (1 - S[:, 1] / S[:, 0]))); m('laserSeedsNR', '%.3f/%.3f and %.3f/%.3f' % (S[0, 2], S[0, 3], S[1, 2], S[1, 3]))
    e0 = dict(ls_['eff0']); e1 = dict(ls_['eff1']); m('laserEpsEff', '%.3f' % float(e1['eps_eff'])); m('laserGcr', '%.3f' % float(e1['g_cr'])); m('laserGammaChange', '%.0f' % (100 * max(abs(float(e1[k]) / float(e0[k]) - 1) for k in ('gamma_x', 'gamma_y', 'gamma_z'))))
    if 'flipL3' in ls_: m('laserFlipImpr', '%.0f' % (100 * (1 - ls_['flipL3'].min() / L0)))
if dg is not None and 'scan' in dg:
    S = dg['scan']; L0 = dg['L0']
    m('digitsAccZero', '%.0f' % (100 * L0[2])); m('digitsAccTrain', '%.0f' % (100 * L0[1])); m('digitsFrontend', '%.0f' % (100 * dg['base'][:, 1].max()))
    m('digitsLrange', '%.1f' % (100 * (S[:, :, 0].max() / S[:, :, 0].min() - 1))); m('digitsAccMin', '%.0f' % (100 * S[:, :, 2].min())); m('digitsAccMax', '%.0f' % (100 * S[:, :, 2].max()))
    if 'rline' in dg: R = dg['rline']; m('digitsRlineAcc', '%.0f--%.0f' % (100 * R[:, 2].min(), 100 * R[:, 2].max()))
    pr = ld('digits_probes.npz')
    if pr is not None:
        for k_, lab in (('Flip', 'flip_r5'), ('EpsHalf', 'eps05'), ('EpsOne', 'eps10'), ('G', 'g09'), ('Tin', 'Tin4')): m('digitsProbe' + k_, '%.0f' % (100 * float(pr['test_acc'][list(pr['labels']).index(lab)])))

dw = ld('digits_wta.npz')
if dw is not None and 'scan' in dw:
    keys = list(dw['keys']); S = dw['scan']; u = dw['unsq']
    i = keys.index('cv_utt'); m('digitsCVrange', '%.1f' % (100 * (S[:, :, i].max() / S[:, :, i].min() - 1))); m('digitsCVzero', '%.3f' % u[i])
    i = keys.index('cv_ce'); m('digitsCErange', '%.1f' % (100 * (S[:, :, i].max() / S[:, :, i].min() - 1)))
    i = keys.index('cv_acc'); m('digitsCVaccMin', '%.0f' % (100 * S[:, :, i].min())); m('digitsCVaccMax', '%.0f' % (100 * S[:, :, i].max())); m('digitsCVaccZero', '%.0f' % (100 * u[i]))
    i = keys.index('test_acc'); m('digitsAccMin', '%.0f' % (100 * S[:, :, i].min())); m('digitsAccMax', '%.0f' % (100 * S[:, :, i].max())); m('digitsAccZero', '%.0f' % (100 * u[i]))
    m('digitsAccTrain', '%.0f' % (100 * u[keys.index('train_acc')])); m('digitsFrameMSE', '%.3f' % u[keys.index('frame_mse')]); m('digitsDelta', '%g' % u[keys.index('delta')])
    if 'rline' in dw:
        R = dw['rline']; i = keys.index('cv_utt'); m('digitsRlineCV', '%.0f' % (100 * (R[:, i].max() / R[0, i] - 1))); m('digitsRlineAcc', '%.0f--%.0f' % (100 * R[:, keys.index('test_acc')].min(), 100 * R[:, keys.index('test_acc')].max()))
    if 'flip' in dw:
        F = dw['flip']; i = keys.index('cv_utt'); m('digitsFlipCV', '%.1f' % (100 * (1 - F[:, i].min() / u[i])))

dr = ld('digits_rre.npz')
if dr is not None and 'grid_m' in dr:
    Gp, Gm = dr['grid_p'], dr['grid_m']; u0 = Gp[0, 0, 0]
    m('digitsRREp', '%.1f' % (100 * (Gp[:, :, 0].max() / Gp[:, :, 0].min() - 1))); m('digitsRREm', '%.1f' % (100 * (Gm[:, :, 0].max() / Gm[:, :, 0].min() - 1)))
    m('digitsRREmGain', '%.1f' % (100 * (1 - Gm[:, :, 0].min() / u0))); i, j = np.unravel_index(Gm[:, :, 0].argmin(), (6, 6)); m('digitsRREmBest', '(%.1f, %.1f)' % (dr['grid'][i], dr['grid'][j]))
    i, j = np.unravel_index(Gp[:, :, 0].argmin(), (6, 6)); m('digitsRREpBest', '(%.1f, %.1f)' % (dr['grid'][i], dr['grid'][j])); m('digitsRREpWorst', '%.1f' % (100 * (Gp[5, :, 0].min() / u0 - 1)))
    m('digitsRREacc', '%.0f--%.0f' % (100 * min(Gp[:, :, 8].min(), Gm[:, :, 8].min()), 100 * max(Gp[:, :, 8].max(), Gm[:, :, 8].max()))); m('digitsRREdth', '%.2f' % dr['dth'])
    mo = dr['motion']; m('digitsMotionMin', '%.0f' % mo[:, 4].min()); m('digitsMotionMax', '%.0f' % mo[:, 4].max())

asr = ld('asr.npz')
if asr is not None and 'best5' in asr:
    u5, b5, fr = asr['unsq5'], asr['best5'], asr['frontend']
    m('asrWERzero', '%.1f' % (100 * u5[:, 1].mean())); m('asrWERbest', '%.1f' % (100 * b5[:, 1].mean())); m('asrWERfront', '%.1f' % (100 * fr.mean()))
    m('asrWERzeroFolds', '/'.join('%.0f' % (100 * x) for x in u5[:, 1])); m('asrWERbestFolds', '/'.join('%.0f' % (100 * x) for x in b5[:, 1])); m('asrWERfrontFolds', '/'.join('%.0f' % (100 * x) for x in fr))
    m('asrWERrel', '%.0f' % (100 * (1 - b5[:, 1].mean() / u5[:, 1].mean()))); m('asrWERsixZero', '%.0f' % (100 * asr['unsq5_all'][:, 1].mean())); m('asrWERsixBest', '%.0f' % (100 * asr['best5_all'][:, 1].mean()))
    b = asr['best']; m('asrBest', '(%.1f, %.1f, %.2f)' % tuple(b)); m('asrBestBranch', 'amplifying' if float(asr['best_s']) < 0 else 'de-amplifying'); m('asrNmax', '%.1f' % b5[:, 3].max())
    S = asr['scan']; L0 = S[0, 0, 0]; m('asrScanRange', '%.1f' % (100 * (1 - S[:, :, 0].min() / L0)))
    Gp, Gm = asr['grid_p'], asr['grid_m']; m('asrGridPImpr', '%.1f' % (100 * (1 - Gp[:, :, 0].min() / L0))); m('asrGridMImpr', '%.1f' % (100 * (1 - Gm[:, :, 0].min() / L0)))
    i, j = np.unravel_index(Gp[:, :, 0].argmin(), (6, 6)); m('asrGridPBest', '(%.1f, %.1f)' % (asr['grid'][i], asr['grid'][j]))
    m('asrWERfoldZero', '%.0f' % (100 * S[0, 0, 1])); m('asrWERfoldMin', '%.0f' % (100 * min(Gp[:, :, 1].min(), Gm[:, :, 1].min()))); m('asrWERfoldMax', '%.0f' % (100 * max(Gp[:, :, 1].max(), Gm[:, :, 1].max())))
    m('asrLfold', '%.5f' % L0); m('asrLfoldBest', '%.5f' % Gm[:, :, 0].min()); m('asrBestR', '%.1f, %.1f' % (b[0], b[1])); m('asrNmaxZero', '%.2f' % u5[:, 3].max())

bs = ld('base.npz')
if bs is not None:
    convs = []
    for nm, U in (('mg', 'MG'), ('narma', 'NARMA'), ('lorenz', 'LORENZ'), ('nce', 'NCE'), ('laser', 'LASER')):
        if f'{nm}_best' not in bs: continue
        E = bs[f'{nm}_eps']; L02 = E[np.isclose(E[:, 0], 0.2)][0, 1]; Lb = E[:, 1].min(); c = bs[f'{nm}_conv']; Ls = c[0, 0]
        m('baseEps' + U, '%.2f' % float(bs[f'{nm}_epsbest'])); m('baseImpr' + U, '%.0f' % (100 * (1 - Lb / L02))); m('sqImpr' + U, '%.0f' % (100 * (1 - Ls / Lb)))
        m('sqBest' + U, '(%.2f, %.2f, %.2f)' % tuple(bs[f'{nm}_best'])); m('sqPd' + U, r'\pi/2' if float(bs[f'{nm}_bestpd']) > 0.5 else '0'); m('sqTotal' + U, '%.0f' % (100 * (1 - Ls / L02)))
        convs.append(abs(c[0, 0] - c[1, 0]) / c[1, 0]); m('sqNmax' + U, '%.1f' % c[1, 2])
    m('sqConvMax', '%.1f' % (100 * max(convs)) if convs else '--'); m('spuriousMG', '26')
    for U in ('MG', 'NARMA', 'LORENZ', 'NCE', 'LASER'):
        for k in ('baseEps', 'baseImpr', 'sqImpr', 'sqBest', 'sqPd', 'sqTotal', 'sqNmax'):
            if k + U not in M: m(k + U, '--')
dv = ld('deriv.npz')
B = chr(92)
if dv is not None:
    f3 = lambda v: '(' + ', '.join('%.2e' % x for x in v) + ')'
    L = [B + 'begin{center}' + B + 'small' + B + 'begin{tabular}{lccc}' + B + 'toprule anchor & fixed $h=10^{-2}$ (projected) & convergent (Richardson) $' + B + 'pm$ error & fine reference' + B + B + ' ' + B + 'midrule']
    for r in dv['rows']:
        L.append('%s %s & %s & %s $%spm$ %s & %s%s%s' % (str(r[0]).replace('_', ' '), '(%.2f, %.2f, %.2f)' % tuple(r[1]), f3(r[2]), f3(r[3]), B, f3(r[4]), f3(r[6]), B, B))
    L.append(B + 'bottomrule' + B + 'end{tabular}' + B + 'end{center}')
    open(os.path.join(PAPER, 'derivtable.tex'), 'w').write(chr(10).join(L))
m('derivTable', B + 'input{derivtable}')
for U in ('MG', 'NARMA', 'LORENZ', 'NCE', 'LASER'):
    if 'sqPd' + U in M and M['sqPd' + U] != '--': M['sqPd' + U] = B + 'ensuremath{' + M['sqPd' + U] + '}'
ab = ld('asr_base.npz')
if ab is not None and 'best5' in ab:
    E = ab['eps']; m('asrBaseEps', '%.2f' % float(ab['epsbest'])); m('asrBaseLmin', '%.5f' % E[:, 1].min()); m('asrBaseLmax', '%.5f' % E[:, 1].max())
    m('asrBaseWERfoldMin', '%.0f' % (100 * E[:, 2].min())); m('asrBaseWERfoldMax', '%.0f' % (100 * E[:, 2].max()))
    m('asrBaseWER', '%.1f' % (100 * ab['base5'][:, 1].mean())); m('asrBaseWERfolds', '/'.join('%.0f' % (100 * x) for x in ab['base5'][:, 1]))
    m('asrSqWER', '%.1f' % (100 * ab['best5'][:, 1].mean())); m('asrSqWERfolds', '/'.join('%.0f' % (100 * x) for x in ab['best5'][:, 1]))
    b = ab['best']; m('asrSqBest', '(%.2f, %.2f)' % (b[0], b[1])); Lb = min(ab['grid_p'][:, :, 0].min(), ab['grid_m'][:, :, 0].min()); L0 = E[np.isclose(E[:, 0], float(ab['epsbest']))][0, 1]
    m('asrSqLimpr', '%.0f' % (100 * (1 - Lb / L0))); m('asrSqPimpr', '%.1f' % (100 * (1 - ab['grid_p'][:, :, 0].min() / L0)))
    m('asrSqNmax', '%.1f' % ab['best5'][:, 3].max())

with open(os.path.join(PAPER, 'numbers.tex'), 'w') as f:
    for k, v in M.items():
        f.write('\\newcommand{\\%s}{%s}\n' % (k, v))
print('\n'.join('%s = %s' % kv for kv in M.items()))
