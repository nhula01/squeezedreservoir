"""Figures and LaTeX macros for the referee-driven revision: fig_refab.pdf, fig_cost.pdf, fig_nonideal.pdf -> paper/figs;
paper/revnumbers.tex, paper/refabtable.tex."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); PAPER = os.path.join(HERE, '..', 'paper'); FIG = os.path.join(PAPER, 'figs')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'axes.titlesize': 8, 'pdf.fonttype': 42, 'figure.dpi': 150})
ld = lambda n: np.load(os.path.join(DATA, n), allow_pickle=True) if os.path.exists(os.path.join(DATA, n)) else None
T = ['mg', 'narma', 'lorenz', 'nce', 'laser']; U = {'mg': 'MG', 'narma': 'NARMA', 'lorenz': 'LORENZ', 'nce': 'NCE', 'laser': 'LASER'}
NM = {'mg': 'Mackey–Glass', 'narma': 'NARMA10', 'lorenz': 'Lorenz-63', 'nce': 'channel eq.', 'laser': 'Santa Fe laser'}
M = {}
W = {3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
def m(k, v): M[k] = v
def lab(ax, s): ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')
def ee(x, d):
    """x in LaTeX scientific notation with d decimals, math-safe."""
    if x == 0: return '0'
    e = int(np.floor(np.log10(abs(x)))); return r'\ensuremath{%.*f\times10^{%d}}' % (d, x / 10 ** e, e)
def sci(x):
    e = int(np.floor(np.log10(x))); c = x / 10 ** e
    return r'\ensuremath{%.0f\times10^{%d}}' % (c, e) if round(c) != 1 else r'\ensuremath{10^{%d}}' % e
pc = lambda a, b: 100 * (1 - a / b)
bs, rf, b2, po = ld('base.npz'), ld('rev_refab.npz'), ld('rev_base2.npz'), ld('rev_post.npz')
KAPPA_HZ = 2 * np.pi * 10e6; OMEGA = 5 * KAPPA_HZ     # kappa/2pi = 10 MHz, omega_a = 5 kappa
T_PASS = 1000 * 2.0 / OMEGA                           # one pass: N_total * T_in

def _bd2(n):
    p = os.path.join(DATA, f'rev_bounds2_{n}.npz'); d = np.load(p, allow_pickle=True) if os.path.exists(p) else None
    return d if d is not None and 'checks' in d.files else None
def _sqf(n):
    p = os.path.join(DATA, f'rev_bounds2_{n}_sqf.npz'); return np.load(p, allow_pickle=True) if os.path.exists(p) else None

def fig_refab():
    """a: relative-precision objective of the paper (tuned drive, squeezing, mapped device, original-box re-fabrication);
    b: measurement-aware objective (rev_bounds2.py): tuned base and squeezed device, re-fabrication in the original box with and
    without squeezing, re-fabrication in the wide box with and without squeezing, joint search."""
    have_b = any(_bd2(n) is not None for n in T)
    fig, axs = plt.subplots(2 if have_b else 1, 1, figsize=(7.0, 5.4 if have_b else 2.6), gridspec_kw=dict(hspace=0.75))
    axs = np.atleast_1d(axs); ax = axs[0]
    cols = ['0.75', '0.55', '#f4a261', '#e76f51', '#8ab17d', '#264653']
    labs = ['base: drive amplitude', 'base: drive amplitude + frequency', 'squeezing (from amplitude base)', 'squeezing (from amplitude + frequency base)',
            'unsqueezed device with the effective parameters', 're-fabrication (6 parameters, original box)']
    w = 0.13; rows = []
    for i, n in enumerate(T):
        Lb = bs[f'{n}_eps'][:, 1].min()
        vals = [Lb, b2[f'{n}_base2'][2], bs[f'{n}_conv'][0, 0], b2[f'{n}_Lsq2'][0][0], po[f'{n}_map2_Lfull'][0], rf[f'{n}_best'][6]]
        rows.append(vals)
        for j, v in enumerate(vals):
            ax.bar(i + (j - 2.5) * w, v / Lb, w, color=cols[j], label=labs[j] if i == 0 else None, edgecolor='k', lw=0.3)
    ax.set_yscale('log'); ax.set_xticks(range(len(T))); ax.set_xticklabels([NM[n] for n in T]); ax.axhline(1, color='k', lw=0.5)
    ax.set_ylabel('loss / amplitude-optimized base'); ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=6)
    ax.set_ylim(0.03, 1.3); lab(ax, 'a')
    if have_b:
        ax = axs[1]; w = 0.12
        cols = ['0.55', '#e76f51', '#9fb8c7', '#5d7f99', '#264653', '#2a9d8f', '#b5838d']
        labs = ['tuned base', 'tuned base + squeezing', 're-fabrication, original box', 're-fabrication, original box + squeezing', 're-fabrication, wide box', 're-fabrication, wide box + squeezing', 'joint search (6 + 4 parameters)']
        for i, n in enumerate(T):
            d = _bd2(n); q = _sqf(n)
            if d is None: continue
            R_ = d['refs']; Lb = R_[0, 0]
            vals = [R_[0, 0], R_[1, 0], R_[2, 0], R_[3, 0], d['ord_best'][10], q['best'][10] if q is not None else np.nan, d['joint_best'][10]]
            for j, v in enumerate(vals):
                if np.isfinite(v): ax.bar(i + (j - 3) * w, v / Lb, w, color=cols[j], label=labs[j] if i == 0 else None, edgecolor='k', lw=0.3)
        ax.set_yscale('log'); ax.set_xticks(range(len(T))); ax.set_xticklabels([NM[n] for n in T]); ax.axhline(1, color='k', lw=0.5)
        ax.set_ylabel('loss / tuned base\n(measurement-aware objective)'); ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=6)
        ax.set_ylim(0.04, 1.3); lab(ax, 'b')
    fig.savefig(os.path.join(FIG, 'fig_refab.pdf'), bbox_inches='tight'); plt.close(fig)
    return np.array(rows)

def numbers_refab(rows):
    PN = ['g', 'kappa', 'gamma', 'omega', 'omega_q', 'eps']
    L = [r'\begin{tabular}{lcccccccc}', r'\toprule', r'task & $g$ & $\kappa$ & $\gamma$ & $\omega_a$ & $\omega_q$ & $\epsilon$ & $\max\langle a^\dagger a\rangle$ & evaluations\\', r'\midrule']
    for i, n in enumerate(T):
        b = rf[f'{n}_best']; ntr = len(rf[f'{n}_trace'])
        L.append(NM[n] + ' & ' + ' & '.join('%.2f' % x for x in b[:6]) + ' & %.2f & %d\\\\' % (b[8], ntr))
        Lb, Lb2, Ls1, Ls2, Lmap, Lrf = rows[i]
        m('rfImpr' + U[n], '%.0f' % pc(Lrf, Lb)); m('bTwoImpr' + U[n], '%.0f' % pc(Lb2, Lb)); m('sqTwoImpr' + U[n], '%.0f' % pc(Ls2, Lb2))
        m('sqTwoTot' + U[n], '%.0f' % pc(Ls2, Lb)); m('sqOneTot' + U[n], '%.0f' % pc(Ls1, Lb)); m('mapImpr' + U[n], '%.0f' % pc(Lmap, Lb2))
        m('recovTwo' + U[n], '%.0f' % (100 * max(0, Lb - Ls2) / (Lb - Lrf)))
        bb = b2[f'{n}_best']; m('bTwoEps' + U[n], '%.2f' % bb[0]); m('bTwoDelta' + U[n], '%+.2f' % bb[1])
        m('sqTwoBest' + U[n], '(%.2f, %.2f, %.2f)' % tuple(bb[3:])); m('sqTwoPd' + U[n], r'\ensuremath{\pi/2}' if bb[2] > 0.5 else '0')
        c = b2[f'{n}_Lsq2']; m('sqTwoConv' + U[n], '%.1f' % (100 * abs(c[0][0] - c[1][0]) / c[1][0]))
        m('sqTwoRes' + U[n], '%.0f' % pc(Ls2, Lmap)); m('rfImprTwo' + U[n], '%.0f' % pc(Lrf, Lb2)); m('sqOverRf' + U[n], '%.1f' % (Ls2 / Lrf))          # squeezed below the mapped device
        m('rfEdge' + U[n], 'yes' if any(np.isclose(b[k], v, rtol=0.03) for k, v in [(0, 1.2), (1, 0.05), (3, 2.0), (4, 2.0), (5, 1.6), (2, 0.02), (2, 0.6)]) else 'no')
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(PAPER, 'refabtable.tex'), 'w').write('\n'.join(L))

def fig_cost():
    gr = ld('rev_grad.npz'); trn = ld('rev_train2.npz') if ld('rev_train2.npz') is not None else ld('rev_train.npz'); task = str(gr['task'])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]; NREP = po['NREP']
    for n in T:
        rb, rs = po[f'{n}_base'], po[f'{n}_sq']
        ax.semilogx(NREP, pc(rs[:, 8], rb[:, 8]), 'o-', ms=3, label=NM[n])
        clean = pc(po[f'{n}_sq_L0'][0], po[f'{n}_base_L0'][0]); ax.plot(NREP[-1] * 2.5, clean, '*', ms=5, color=ax.lines[-1].get_color())
    ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('passes per setting $N_{\\rm rep}$'); ax.set_ylabel('measured loss reduction by squeezing (%)')
    ax.set_ylim(-60, 100); ax.legend(frameon=False, fontsize=5, loc='upper left', ncol=1, bbox_to_anchor=(0.0, 0.93)); lab(ax, 'a')
    sec = ax.secondary_xaxis('top', functions=(lambda N: 2 * N * T_PASS, lambda t: t / (2 * T_PASS))); sec.set_xlabel('s per evaluation ($\\kappa/2\\pi=10$ MHz)', fontsize=6.5)
    ax = axs[1]; H = gr['H']; an = 'start'; g = gr[f'{an}_gtrue']; G = gr[f'{an}_G']
    for b, N in enumerate(gr['NREP']):
        err = np.sqrt(np.mean(np.sum((G[:, b] - g) ** 2, -1), -1)) / np.linalg.norm(g)
        ax.loglog(H, err, 'o-', ms=3, label='$N_{\\rm rep}=10^{%d}$' % round(np.log10(N)))
    ax.axhline(1, color='k', lw=0.5, ls=':'); ax.set_xlabel('finite-difference step $h$'); ax.set_ylabel('rms gradient error / $|\\nabla\\mathcal{L}|$'); ax.legend(frameon=False); lab(ax, 'b')
    ax = axs[2]; L0 = None
    import re
    for key in sorted(k for k in trn.files if k.endswith('_trace')):
        mth, lg = key.split('_')[0], int(key.split('_')[1]); ni = '_ni_' in key
        tr = trn[key]; passes = np.arange(1, len(tr) + 1) * 2 * 10 ** lg
        best_clean = tr[:, 5]  # clean loss at every evaluated setting
        run = np.minimum.accumulate(tr[:, 4]); idx = [int(np.argmin(tr[:i + 1, 4])) for i in range(len(tr))]
        y = best_clean[idx]                                  # clean loss of the setting the optimizer currently believes best
        if L0 is None: L0 = b2[f'{task}_base2'][2]
        c = {5: 'C0', 6: 'C1', 7: 'C2'}[lg]
        ax.semilogx(passes * T_PASS, y / L0, color='k' if ni else c, ls='-' if mth == 'gd' else '--', lw=0.8, alpha=0.8)
    Ls = b2[f'{task}_Lsq2'][0][0]; ax.axhline(Ls / L0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('measurement time (s, $\\kappa/2\\pi=10$ MHz)'); ax.set_ylabel('clean loss / base'); lab(ax, 'c')
    ax.plot([], [], 'k-', lw=0.8, label='FD gradient descent'); ax.plot([], [], 'k--', lw=0.8, label='SPSA')
    for lg, c in ((5, 'C0'), (6, 'C1'), (7, 'C2')): ax.plot([], [], color=c, label='$10^{%d}$' % lg)
    ax.plot([], [], color='k', label='$10^6$, all non-idealities'); ax.legend(frameon=False, fontsize=5, loc='lower left')
    fig.savefig(os.path.join(FIG, 'fig_cost.pdf'), bbox_inches='tight'); plt.close(fig)
    return gr, trn, task

def numbers_cost(gr, trn, task):
    m('costTask', NM[task]); NREP = po['NREP']
    # precision needed for 1%: median sigma_rel scales as N^{-1/2}
    s = np.array([po[f'{n}_{k}'][:, 1] for n in T for k in ('base', 'sq')]); Nneed = (s * np.sqrt(NREP) / 0.01) ** 2
    m('NoneLo', sci(np.min(np.median(Nneed, 1)))); m('NoneHi', sci(np.max(np.median(Nneed, 1))))
    m('NoneMed', '%.1f' % (np.median(Nneed) / 1e6))
    m('tPass', '%.1f' % (T_PASS * 1e6)); m('tEvalSix', '%.0f' % (2e6 * T_PASS))
    for n in T:
        rb, rs = po[f'{n}_base'], po[f'{n}_sq']
        for i, N in enumerate(NREP):
            m('gainN' + W[round(np.log10(N))] + U[n], '%.0f' % pc(rs[i, 8], rb[i, 8]))
        m('gainClean' + U[n], '%.0f' % pc(po[f'{n}_sq_L0'][0], po[f'{n}_base_L0'][0]))
    for an in ('start', 'mid'):
        g = gr[f'{an}_gtrue']; G = gr[f'{an}_G']; A = an.capitalize()
        m('gTrue' + A, '(' + ', '.join(ee(x, 2) for x in g) + ')')
        for b, N in enumerate(gr['NREP']):
            err = np.sqrt(np.mean(np.sum((G[:, b] - g) ** 2, -1), -1)) / np.linalg.norm(g); ib = err.argmin()
            m('gErrN' + W[round(np.log10(N))] + A, '%.0f' % (100 * err[ib])); m('gHN' + W[round(np.log10(N))] + A, '%g' % gr['H'][ib])
            sgn = np.mean(np.sign(G[ib, b]) == np.sign(g), 0)
            m('gSignN' + W[round(np.log10(N))] + A, '(' + ', '.join('%.0f' % (100 * x) for x in sgn) + ')')
        m('sigLrel' + A, '(' + ', '.join(ee(x, 1) for x in gr[f'{an}_sigL'] / gr[f'{an}_L0']) + ')')
        m('Lthree' + A, '(' + ', '.join('%.2g' % x for x in gr[f'{an}_L3']) + ')')
        hstar = (3 * gr[f'{an}_sigL'][:, None] / np.maximum(np.abs(gr[f'{an}_L3'])[None, :], 1e-12)) ** (1 / 3)
        m('hStar' + A, '(' + ', '.join('%.2f' % np.median(h) for h in hstar) + ')')
    L0 = b2[f'{task}_base2'][2]; Ls = b2[f'{task}_Lsq2'][0][0]
    for mth in ('gd', 'spsa'):
        for lg in (5, 6, 7):
            ks = [k for k in trn.files if k.startswith(f'{mth}_{lg}_') and k.endswith('_Lfinal') and '_ni_' not in k]
            vals = [pc(float(trn[k]), L0) for k in ks]
            m(f'tr{mth.upper()}N' + W[lg], (('%.0f' % min(vals)) if round(min(vals)) == round(max(vals)) else '%.0f--%.0f' % (min(vals), max(vals))) if vals else '--')
            m('trTime' + W[lg].capitalize(), '%.1f' % (48 * 2 * 10 ** lg * T_PASS / 60))
        ks = [k for k in trn.files if k.startswith(f'{mth}_6_') and k.endswith('_ni_Lfinal_ni')]
        vals = [pc(float(trn[k]), L0) for k in ks]; m(f'tr{mth.upper()}NI', (('%.0f' % min(vals)) if round(min(vals)) == round(max(vals)) else '%.0f--%.0f' % (min(vals), max(vals))) if vals else '--')
    m('trIdeal', '%.0f' % pc(Ls, L0)); m('trBudget', '48')

def fig_nonideal():
    ni = ld('rev_nonideal.npz')
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.6]))
    ax = axs[0]; ETAS = [1.0, 0.9, 0.7, 0.5]
    for n in T:
        Lb = ni[f'{n}_Lb']; ax.plot(ETAS, pc(ni[f'{n}_eta'], Lb), 'o-', ms=3, label=NM[n])
        ax.plot([0.7], [pc(ni[f'{n}_reopt_7'][4], Lb)], 's', mfc='none', ms=4, color=ax.lines[-1].get_color())
    ax.set_xlabel('injection efficiency $\\eta_{\\rm inj}$'); ax.set_ylabel('loss reduction by squeezing (%)'); ax.invert_xaxis(); ax.set_ylim(-5, 80); ax.legend(frameon=False, fontsize=5.5, loc='center right', bbox_to_anchor=(1.0, 0.62)); lab(ax, 'a')
    ax = axs[1]; cats = ['ideal', '$\\sigma_\\phi=0.2$', 'drift\n(moderate)', 'drift\n(large)', 'readout\n$10^6$', 'all at once']; w = 0.15
    for i, n in enumerate(T):
        Lb = ni[f'{n}_Lb']
        v = [pc(ni[f'{n}_Ls'], Lb), pc(ni[f'{n}_sig'][2], Lb), pc(ni[f'{n}_drift_moderate'][:, 0].mean(), ni[f'{n}_drift_moderate'][:, 1].mean()),
             pc(ni[f'{n}_drift_large'][:, 0].mean(), ni[f'{n}_drift_large'][:, 1].mean()), pc(ni[f'{n}_ro'][:, 0].mean(), ni[f'{n}_ro'][:, 1].mean()),
             pc(ni[f'{n}_comb'][:, 0].mean(), ni[f'{n}_comb'][:, 1].mean())]
        ax.bar(np.arange(len(cats)) + (i - 2) * w, v, w, label=NM[n], edgecolor='k', lw=0.3)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=6.5); ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('loss reduction by squeezing (%)'); lab(ax, 'b')
    fig.savefig(os.path.join(FIG, 'fig_nonideal.pdf'), bbox_inches='tight'); plt.close(fig)
    for n in T:
        Lb = ni[f'{n}_Lb']
        m('niIdeal' + U[n], '%.0f' % pc(ni[f'{n}_Ls'], Lb))
        for j, e in enumerate([10, 9, 7, 5]): m('niEta' + W[e].capitalize() + U[n], '%.0f' % pc(ni[f'{n}_eta'][j], Lb))
        m('niReoptSeven' + U[n], '%.0f' % pc(ni[f'{n}_reopt_7'][4], Lb))
        for j, s in enumerate(['A', 'B', 'C', 'D']): m('niSig' + s + U[n], '%.0f' % pc(ni[f'{n}_sig'][j], Lb))
        for dn in ('moderate', 'large'):
            d = ni[f'{n}_drift_{dn}']; m('niDrift' + dn.capitalize() + U[n], '%.0f' % pc(d[:, 0].mean(), d[:, 1].mean()))
        m('niRO' + U[n], '%.0f' % pc(ni[f'{n}_ro'][:, 0].mean(), ni[f'{n}_ro'][:, 1].mean()))
        m('niComb' + U[n], '%.0f' % pc(ni[f'{n}_comb'][:, 0].mean(), ni[f'{n}_comb'][:, 1].mean()))

def perf_and_eff_tables():
    """Held-out NRMSE table (tuned base, squeezed, re-fabricated, software baselines) and effective parameters at the tuned optima."""
    rv, lz = ld('review.npz'), ld('laser.npz')
    rowsL = [r'\begin{tabular}{lccccc}', r'\toprule', r'Device or method & Mackey--Glass & NARMA10 & Lorenz-63 & channel eq. & laser\\', r'\midrule']
    def base_of(n):
        B = lz['base'] if n == 'laser' else rv[f'base_{n}']
        return {str(r[0]): float(r[2]) for r in B}
    cols = {n: base_of(n) for n in T}
    rows = [('tuned drive, unsqueezed', lambda n: b2[f'{n}_base2'][3]), ('tuned drive, squeezed', lambda n: b2[f'{n}_Lsq2'][0][1]),
            ('re-fabricated, original box', lambda n: rf[f'{n}_best'][7]), ('\\quad + squeezing', lambda n: float(M.get('nrRfSq' + U[n], 'nan'))),
            ('re-fabricated, wide box$^\\dagger$', lambda n: _bd2(n)['ord_best'][11] if _bd2(n) is not None else np.nan),
            ('\\quad + squeezing$^\\dagger$', lambda n: _sqf(n)['best'][11] if _sqf(n) is not None else np.nan), ('linear regression, 10 delays', lambda n: cols[n]['ols10']),
            ('quadratic regression, 10 delays', lambda n: cols[n]['quad10']), ('100-node echo-state network', lambda n: cols[n]['esn100'])]
    for name, fn in rows: rowsL.append(name + ' & ' + ' & '.join(('%.3f' % fn(n)) if np.isfinite(fn(n)) else '--' for n in T) + r'\\')
    rowsL += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'perftable.tex'), 'w').write('\n'.join(rowsL))
    for n in T:
        m('nrBaseTwo' + U[n], '%.3f' % b2[f'{n}_base2'][3]); m('nrSqTwo' + U[n], '%.3f' % b2[f'{n}_Lsq2'][0][1]); m('nrRf' + U[n], '%.3f' % rf[f'{n}_best'][7])
        m('nrOLS' + U[n], '%.3f' % cols[n]['ols10']); m('nrESN' + U[n], '%.3f' % cols[n]['esn100'])
    import sys; sys.path.insert(0, HERE)
    from rev_common import squeezed_point2, effective_parameters
    keys = [('eps_eff', r'$\epsilon_{\rm eff}$'), ('Omega', r'$\Omega_a$'), ('g_co', r'$g\cosh r$'), ('g_cr', r'$g\sinh r$'),
            ('gamma_x', r'$\gamma_x^{\rm eff}$'), ('gamma_y', r'$\gamma_y^{\rm eff}$'), ('gamma_z', r'$\gamma_z^{\rm eff}$'), ('VarQ', r'$\mathrm{Var}(Q)$'), ('VarP', r'$\mathrm{Var}(P)$')]
    tasks = ['narma', 'lorenz', 'nce', 'laser']; E = {}
    for n in tasks:
        dv, th = squeezed_point2(n); E[n] = (effective_parameters(dv, np.zeros(3)), effective_parameters(dv, th))
    L = [r'\begin{tabular}{l' + 'cc' * len(tasks) + '}', r'\toprule', ' & ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % NM[n].replace('–', '--') for n in tasks) + r'\\',
         ' & ' + ' & '.join(['base & sq.'] * len(tasks)) + r'\\', r'\midrule']
    L.append(r'$(r,\re,\dth)$ & ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % M['sqTwoBest' + U[n]] for n in tasks) + r'\\')
    for k, lab_ in keys: L.append(lab_ + ' & ' + ' & '.join('%.3f & %.3f' % (E[n][0][k], E[n][1][k]) for n in tasks) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'efftable2.tex'), 'w').write('\n'.join(L))
    for n in tasks: m('gcrTwo' + U[n], '%.2f' % E[n][1]['g_cr']); m('VarQTwo' + U[n], '%.2f' % E[n][1]['VarQ']); m('VarPTwo' + U[n], '%.2f' % E[n][1]['VarP'])


def asr_numbers():
    a = ld('rev_asr2.npz'); W = a['wer5']; sel = a['sel']
    m('asrTwoWERfab', '%.1f' % (100 * W[0, :, 1].mean())); m('asrTwoWERbase', '%.1f' % (100 * W[1, :, 1].mean())); m('asrTwoWERsq', '%.1f' % (100 * W[2, :, 1].mean()))
    f = lambda w: ', '.join('%.0f' % (100 * x) for x in w)
    m('asrTwoFoldsBase', f(W[1, :, 1])); m('asrTwoFoldsSq', f(W[2, :, 1])); m('asrTwoFoldsFab', f(W[0, :, 1]))
    m('asrTwoEps', '%.1f' % sel[0]); m('asrTwoDelta', '%+.2f' % sel[1]); m('asrTwoPd', r'\ensuremath{\pi/2}' if sel[2] > 0.5 else '0')
    m('asrTwoBest', '(%.2f, %.2f, %.2f)' % tuple(sel[3:])); m('asrTwoLimpr', '%.1f' % pc(W[2, 0, 0], W[1, 0, 0]))
    m('asrTwoConv', '%.1f' % (100 * abs(a['conv'][0] - W[2, 0, 0]) / a['conv'][0]))

def snr_numbers():
    s = ld('rev_snr.npz'); NREP = s['NREP']; i6 = list(NREP).index(1e6)
    for n in T:
        rb, rs = s[f'{n}_base'], s[f'{n}_sq']
        m('snrGainSix' + U[n], '%.0f' % pc(rs[i6, 3], rb[i6, 3])); m('snrHeldSix' + U[n], '%.0f' % pc(rs[i6, 4], rb[i6, 4]))
    a = s['sigL_start']; m('snrSigL', '(' + ', '.join(ee(x, 1) for x in a[:, 1] / a[0, 2]) + ')')

def train2_numbers():
    t = ld('rev_train2.npz'); L0 = b2[f'{str(t["task"])}_base2'][2]; ok = 0; tot = 0; allv = []
    for mth in ('gd', 'spsa'):
        for lg in (5, 6, 7):
            v = [pc(float(t[k]), L0) for k in t.files if k.startswith(f'{mth}_{lg}_') and k.endswith('_Lfinal') and '_ni_' not in k]
            m(f'trTwo{mth.upper()}N' + W[lg], (('%.0f' % min(v)) if round(min(v)) == round(max(v)) else '%.0f--%.0f' % (min(v), max(v))) if v else '--'); allv += v
        v = [pc(float(t[k]), L0) for k in t.files if k.startswith(f'{mth}_6_') and k.endswith('_ni_Lfinal')]
        m(f'trTwo{mth.upper()}NI', (('%.0f' % min(v)) if round(min(v)) == round(max(v)) else '%.0f--%.0f' % (min(v), max(v))) if v else '--'); allv += v
    allv = np.array(allv); gain = pc(b2[f'{str(t["task"])}_Lsq2'][0][0], L0)
    m('trTwoRuns', '%d' % len(allv)); m('trTwoGood', '%d' % np.sum(allv >= 0.5 * gain)); m('trTwoMin', '%.0f' % allv.min())


def lo_numbers():
    lo, dg = ld('rev_lo.npz'), ld('rev_lo_diag.npz'); NREP = list(lo['NREP'])
    def gain(n, k, N, tag=''):
        i = NREP.index(N); b = min(lo[f'{n}{tag}_{bb}'][i, 1] for bb in ('base_QP', 'base_QQ', 'base_PP'))
        return pc(lo[f'{n}{tag}_{k}'][i, 1], b)
    names = [('sq_QPvac', r'$(Q,P)$, vacuum input noise'), ('sq_QP', r'$(Q,P)$, squeezed input noise'),
             ('sq_SA', r'$(X_{\phi_{\rm sq}},X_{\phi_{\rm sq}+\pi/2})$'), ('sq_SS', r'$(X_{\phi_{\rm sq}},X_{\phi_{\rm sq}})$')]
    tk = ['narma', 'lorenz', 'nce', 'laser']
    L = [r'\begin{tabular}{l' + 'cc' * len(tk) + '}', r'\toprule', 'LO angles of the two settings & ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % NM[n].replace('–', '--') for n in tk) + r'\\',
         ' & ' + ' & '.join([r'$10^6$ & $10^7$'] * len(tk)) + r'\\', r'\midrule']
    for k, lab_ in names:
        L.append(lab_ + ' & ' + ' & '.join('%.0f & %.0f' % (gain(n, k, 1e6), gain(n, k, 1e7)) for n in tk) + r'\\')
    L.append(r'\midrule')
    L.append(r'$V_{\rm in}$ at $(Q,\ P,\ \phi_{\rm sq},\ \phi_{\rm sq}+\pi/2)$ & ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % ', '.join('%.2f' % v for v in dg[f'{n}_V']) for n in tk) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'lotable.tex'), 'w').write('\n'.join(L))
    for n in tk:
        m('loQPvac' + U[n], '%.0f' % gain(n, 'sq_QPvac', 1e6)); m('loQP' + U[n], '%.0f' % gain(n, 'sq_QP', 1e6)); m('loSS' + U[n], '%.0f' % gain(n, 'sq_SS', 1e6))
        m('loSA' + U[n], '%.0f' % gain(n, 'sq_SA', 1e6)); m('loQPni' + U[n], '%.0f' % gain(n, 'sq_QP', 1e6, '_ni'))
        v = dg[f'{n}_V']; m('VinQ' + U[n], '%.2f' % v[0]); m('VinP' + U[n], '%.2f' % v[1]); m('VinS' + U[n], '%.2f' % v[2]); m('VinA' + U[n], '%.2f' % v[3])
        i6 = list(dg['NN']).index(1e6); r = dg[f'{n}_QP']; L0 = dg[f'{n}_QP_L0']
        m('diagFull' + U[n], '%.2f' % (r[i6, 1] / L0[0])); m('diagCav' + U[n], '%.2f' % (r[i6, 2] / L0[0]))
        m('cavOnlyClean' + U[n], '%.3g' % L0[1]); m('cavOnlySSClean' + U[n], '%.3g' % dg[f'{n}_SS_L0'][1])
        b = dg[f'{n}_baseQP']; m('cavOnlyGain' + U[n], '%.0f' % pc(r[i6, 3], b[i6, 3])); m('cavOnlySSGain' + U[n], '%.0f' % pc(dg[f'{n}_SS'][i6, 3], b[i6, 3]))
    m('dBsqNCE', '%.1f' % (-10 * np.log10(dg['nce_V'][2])))

# ---------------------------------------------------------------- second review round
PNL = [r'$g$', r'$\kappa$', r'$\gamma$', r'$\omega_a$', r'$\omega_q$', r'$\epsilon$']
PN6 = ['g', 'kappa', 'gamma', 'omega_a', 'omega_q', 'eps']
LOW_W = np.array([0.005, 0.005, 0.002, 0.0, 0.0, 0.001]); HIW_W = np.array([3.0, 3.0, 2.0, 5.0, 5.0, 5.0]); LOG_W = np.array([1, 1, 1, 0, 0, 1], bool)
def _u(p):
    p = np.asarray(p, float); u = (p - LOW_W) / (HIW_W - LOW_W); u[LOG_W] = np.log(p[LOG_W] / LOW_W[LOG_W]) / np.log(HIW_W[LOG_W] / LOW_W[LOG_W]); return u
def best_of(tr):
    tr = np.asarray(tr); ok = tr[:, 12] <= 2.5; return tr[np.where(ok)[0][tr[ok, 10].argmin()]]

def refabsq_numbers():
    """Squeezing on the re-fabricated optimum of the original box (rev_refab_sq.py, rev_refab_sq14.py) -> paper/refabsqnumbers.tex."""
    sq, s14 = ld('rev_refab_sq.npz'), ld('rev_refab_sq14.npz'); Q = {}
    if sq is None: return
    rows = []
    for n in T:
        if n in ('mg', 'lorenz') and s14 is not None and f'{n}_best' in s14.files:
            Lrf, Ls, th = s14[f'{n}_Lrf'][1][0], s14[f'{n}_Lsq'][1][0], s14[f'{n}_best']; nc = 18; nr = s14[f'{n}_Lsq'][0][1]
        else:
            Lrf, Ls, th = rf[f'{n}_bestconv'][0], sq[f'{n}_Lsq'][1][0], sq[f'{n}_best']; nc = 14; nr = sq[f'{n}_Lsq'][0][1]
        Q['rfsqImpr' + U[n]] = '%.0f' % max(0, pc(Ls, Lrf)); Q['rfsqBest' + U[n]] = '(%.2f, %.2f, %.2f)' % tuple(th[1:])
        M['nrRfSq' + U[n]] = '%.3f' % nr
        rows.append((n, th, Lrf, Ls, nc))
    Q['rfsqNmaxNCE'] = '%.2f' % rf['nce_best'][8]; Q['rfsqGNCE'] = '%.2f' % rf['nce_best'][0]; Q['rfsqEvals'] = '%d' % int(sq['narma_evals'])
    with open(os.path.join(PAPER, 'refabsqnumbers.tex'), 'w') as f:
        f.write('% Squeezing on the re-fabricated device of the original search box (generated by make_rev.py from rev_refab_sq.npz, rev_refab_sq14.npz)\n')
        for k, v in Q.items(): f.write('\\newcommand{\\%s}{%s}\n' % (k, v))
    L = [r'\begin{tabular}{lccccc}', r'\toprule', r'task & $\phi_d$ & $(r,\re,\dth)$ & $\Lc$ re-fabricated & $\Lc$ + squeezing & $N_c$\\', r'\midrule']
    for n, th, Lrf, Ls, nc in rows:
        L.append(NM[n].replace('–', '--') + r' & %s & (%.2f, %.2f, %.2f) & %s & %s (%+.0f\%%) & %d\\' % (r'$\pi/2$' if th[0] > 0.5 else '0', *th[1:], sci3(Lrf), sci3(Ls), -max(0, pc(Ls, Lrf)), nc))
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'refabsqtable.tex'), 'w').write('\n'.join(L))

def sci3(x):
    e = int(np.floor(np.log10(abs(x)))); return r'\ensuremath{%.2f\times10^{%d}}' % (x / 10 ** e, e)

def bounds_rows():
    """Measurement-aware wide-box study (rev_bounds2.py, rev_bounds2_final.py); losses at N_c = 10 and at the checked cutoff."""
    R = {}
    for n in T:
        d = _bd2(n); q = _sqf(n)
        if d is None: continue
        c = lambda a: a[-1][0]
        R[n] = dict(d=d, q=q, ord=d['ord_best'], seq=d['seq_best'], mat=d['mat_best'], joint=d['joint_best'], refs=d['refs'],
                    cord=c(d['chk_ord']), cseq=c(d['chk_seq']), cmat=c(d['chk_mat']), cjoint=c(d['chk_joint']),
                    sqf=q['best'] if q is not None else None, csqf=c(q['chk']) if q is not None else np.nan, csqf0=c(q['chk0']) if q is not None else np.nan,
                    Lmap0=d['map_L0'][0], Lmapre=best_of(d['map_trace'])[10], Lord300=d['ord_best500'][10],
                    face=[PN6[k] for k, x in enumerate(_u(d['ord_best'][:6])) if min(x, 1 - x) < 0.02],
                    ordtr=d['ord_trace'], jtr=d['joint_trace'], nseq=int(d['seq_evals']), rel=d['rel'])
    return R

def bounds_numbers(R):
    for n, r in R.items():
        u = U[n]; refs = r['refs']; Lt, Lts, Lo, Los = refs[:, 0]
        m('bdOrdVsTuned' + u, '%.0f' % pc(r['ord'][10], Lt)); m('bdOrdVsOld' + u, '%.0f' % pc(r['ord'][10], Lo))
        m('bdTunedSqImpr' + u, '%.0f' % pc(Lts, Lt)); m('bdOldSqImpr' + u, '%.0f' % max(0.0, pc(Los, Lo)))
        m('bdTunedSqOverRf' + u, '%.2f' % (Lts / r['ord'][10])); m('bdOldSqOverRf' + u, '%.2f' % (Los / r['ord'][10]))
        m('bdSqfImpr' + u, '%.1f' % max(0.0, pc(r['csqf'], r['csqf0']))); m('bdSeqImpr' + u, '%.1f' % pc(r['seq'][10], r['Lord300']))
        m('bdMatImpr' + u, '%.1f' % max(0.0, pc(r['cmat'], r['cord']))); m('bdJointImpr' + u, '%+.1f' % (-pc(r['cjoint'], r['cord'])))
        m('bdJointL' + u, sci3(r['joint'][10])); m('bdOrdL' + u, sci3(r['ord'][10]))
        m('bdMapRe' + u, '%+.0f' % (100 * (min(r['Lmapre'], r['Lmap0']) / r['ord'][10] - 1)))
        if r['sqf'] is not None:
            m('bdSqfBest' + u, '(%.2f, %.2f, %.2f)' % tuple(r['sqf'][6:9]))
        m('bdJointSq' + u, '(%.2f, %.2f, %.2f)' % tuple(r['joint'][6:9]))
        for k, nm_ in enumerate(PN6): m('bd' + nm_.replace('_', '').capitalize() + u, '%.3g' % r['ord'][k])
        m('bdNmax' + u, '%.2f' % r['ord'][12]); m('bdFace' + u, ', '.join(r['face']) or 'none')
        m('nrBdOrd' + u, '%.3f' % r['ord'][11])
        if r['sqf'] is not None: m('nrBdSqf' + u, '%.3f' % r['sqf'][11])
        m('bdRelOrd' + u, sci3(r['rel'][0][0]))
    m('bdNfaces', str(sum(1 for r in R.values() if r['face'])))
    sq_all = [float(M['bdSqfImpr' + U[n]]) for n in R if 'bdSqfImpr' + U[n] in M]
    if sq_all: m('bdSqfMax', '%.1f' % max(sq_all))
    mt = [float(M['bdMatImpr' + U[n]]) for n in R]; m('bdMatMax', '%.1f' % max(mt))
    m('bdJointGainMax', '%.1f' % max(max(0.0, pc(r['cjoint'], r['cord'])) for r in R.values()))
    jt = [float(M['bdJointImpr' + U[n]]) for n in R]; m('bdJointMin', '%+.1f' % min(jt)); m('bdJointMax', '%+.1f' % max(jt))
    m('bdEvals', '506')
    for n in R:
        jz, jl = ld(f'rev_bounds2_{n}_jointzero.npz'), ld(f'rev_bounds2_{n}_jointlocal.npz')
        if jz is not None: m('bdJointShare' + U[n], '%.0f' % pc(R[n]['joint'][10], jz['L'][0]))
        if jl is not None:
            m('bdJointLocal' + U[n], sci3(jl['best'][10])); m('bdJointLocalVs' + U[n], '%.1f' % pc(R[n]['joint'][10], jl['best'][10]))
            m('bdJointLocalVsOrd' + U[n], '%.1f' % pc(jl['best'][10], R[n]['ord'][10]))
    bu = ld('rev_bounds2_lorenz_budget.npz')
    if bu is not None:
        b0 = bu['ord_best']; sq = bu['sq']; k = int(np.argmin(sq[:, 4])); c = bu['chk20']
        m('buOrdL', sci3(b0[10])); m('buOrdNmax', '%.2f' % b0[12]); m('buOrdEps', '%.2f' % b0[5]); m('buOrdVs', '%.0f' % pc(b0[10], R['lorenz']['cord']) if 'lorenz' in R else '')
        m('buSqL', sci3(sq[k, 4])); m('buSqImpr', '%.1f' % pc(sq[k, 4], b0[10])); m('buSqRe', '%.2f' % sq[k, 2]); m('buSqR', '%.2f' % sq[k, 1])
        m('buChkImpr', '%.1f' % pc(c[1][0], c[0][0]))
    # the relative-precision objective in the wide box (rev_bounds.py, ordinary stage)
    for n in ('narma', 'nce'):
        d = ld(f'rev_bounds_{n}.npz')
        if d is None or 'ord_trace' not in d.files: continue
        b = best_of(d['ord_trace']); m('relWideEps' + U[n], '%.3f' % b[5]); m('relWideG' + U[n], '%.2f' % b[0]); 
        m('relWideNmax' + U[n], sci3(b[12]))
        m('relWideL' + U[n], sci3(b[10])); m('relWideVsOld' + U[n], '%.0f' % pc(b[10], rf[f'{n}_best'][6]))
    d = ld('rev_bounds_narma.npz')
    if d is not None and 'sanity_best' in d.files:
        m('bdSanityG', '%.3f' % d['sanity_best'][0]); m('bdSanityL', sci3(d['sanity_best'][3]))

def bounds_table(R):
    L = [r'\begin{tabular}{l' + 'c' * 9 + '}', r'\toprule',
         r'task & ' + ' & '.join(PNL) + r' & $\max\langle a^\dagger a\rangle$ & $\phi_d$, $(r,\re,\dth)$ on it & joint: $(r,\re,\dth,\phi_d)$\\', r'\midrule']
    for n, r in R.items():
        o = r['ord']; q = r['sqf']; j = r['joint']
        sq = (r'%s, (%.2f, %.2f, %.2f)' % (r'$\pi/2$' if q[9] > 0.5 else '0', *q[6:9])) if q is not None else '--'
        L.append(NM[n].replace('–', '--') + ' & ' + ' & '.join('%.3g' % o[k] for k in range(6)) + ' & %.2f & %s & (%.2f, %.2f, %.2f, %.2f)\\\\' % (o[12], sq, *j[6:10]))
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'boundstable.tex'), 'w').write('\n'.join(L))
    L = [r'\begin{tabular}{lcccccccc}', r'\toprule', r'task & tuned base & tuned + sq. & old box & old box + sq. & wide box & wide box + sq. & matched & joint\\', r'\midrule']
    for n, r in R.items():
        refs = r['refs'][:, 0]; Lo = r['ord'][10]
        f = lambda x: sci3(x) if np.isfinite(x) else '--'
        L.append(NM[n].replace('–', '--') + ' & ' + ' & '.join(f(x) for x in refs) + ' & %s & %s & %s & %s\\\\' % (
            f(Lo), f(r['sqf'][10] if r['sqf'] is not None else np.nan), f(r['mat'][10]), f(r['joint'][10])))
    L += [r'\midrule', r'\multicolumn{9}{l}{\footnotesize checked cutoff ($N_c=14$, or 18 near the photon budget): wide box / + sq. / joint}\\']
    for n, r in R.items():
        L.append(NM[n].replace('–', '--') + r' & \multicolumn{4}{c}{} & %s & %s & %s & %s\\' % (sci3(r['cord']), sci3(r['csqf']) if np.isfinite(r['csqf']) else '--', sci3(r['cmat']), sci3(r['cjoint'])))
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'boundstable2.tex'), 'w').write('\n'.join(L))

def fig_bounds(R):
    fig, axs = plt.subplots(1, len(R), figsize=(7.2, 1.9), gridspec_kw=dict(wspace=0.5))
    axs = np.atleast_1d(axs)
    for ax, (n, r) in zip(axs, R.items()):
        pen = lambda tr: np.minimum.accumulate(np.where(tr[:, 12] <= 2.5, tr[:, 10], np.inf))
        ref = r['ord'][10]
        ax.plot(np.arange(1, len(r['ordtr']) + 1), pen(r['ordtr']) / ref, color='0.4', lw=1, label='ordinary (6 par.)')
        ax.plot(np.arange(1, len(r['jtr']) + 1), pen(r['jtr']) / ref, color='#b5838d', lw=1, label='joint (10 par.)')
        ax.plot([300, r['nseq']], [r['Lord300'] / ref, r['seq'][10] / ref], 'o-', color='#e76f51', ms=2.5, lw=1, label='sequential')
        if r['sqf'] is not None: ax.plot([len(r['ordtr']), len(r['ordtr']) + 206], [1, r['sqf'][10] / ref], 's-', color='#2a9d8f', ms=2.5, lw=1, label='squeezing on final')
        ax.axhline(1, color='k', lw=0.4, ls=':'); ax.set_title(NM[n], fontsize=7); ax.set_xlabel('evaluations', fontsize=7)
        ax.set_ylim(0.85, 3.0)
    axs[0].set_ylabel('best loss / wide-box optimum'); axs[0].legend(frameon=False, fontsize=4.5)
    fig.savefig(os.path.join(FIG, 'figS_bounds.pdf'), bbox_inches='tight'); plt.close(fig)

def seeds_numbers():
    sd = ld('rev_seeds.npz')
    if sd is None: return
    L = [r'\begin{tabular}{lcccc}', r'\toprule', r'task & gain, seed 0 (paper) & gain, re-optimized, seeds 1--8 & range & gain, settings transferred\\', r'\midrule']
    for n in ('narma', 'nce'):
        ks = sorted(int(k.split('_')[1]) for k in sd.files if k.startswith(n + '_') and k.endswith('_reopt'))
        if not ks: continue
        R_ = np.array([sd[f'{n}_{s}_reopt'] for s in ks]); A = np.array([sd[f'{n}_{s}_trans'] for s in ks])
        g0 = pc(b2[f'{n}_Lsq2'][0][0], b2[f'{n}_base2'][2])
        m('seedGain' + U[n], '%.0f' % (100 * R_[:, 3].mean())); m('seedSd' + U[n], '%.0f' % (100 * R_[:, 3].std(ddof=1)))
        m('seedMin' + U[n], '%.0f' % (100 * R_[:, 3].min())); m('seedMax' + U[n], '%.0f' % (100 * R_[:, 3].max()))
        m('seedTrans' + U[n], '%.0f' % (100 * A[:, 3].mean())); m('seedTransSd' + U[n], '%.0f' % (100 * A[:, 3].std(ddof=1))); m('seedN', W.get(len(ks), str(len(ks))))
        L.append(NM[n] + r' & %.0f\%% & $%.0f\pm%.0f$\%% & %.0f--%.0f\%% & $%.0f\pm%.0f$\%%\\' % (g0, 100 * R_[:, 3].mean(), 100 * R_[:, 3].std(ddof=1), 100 * R_[:, 3].min(), 100 * R_[:, 3].max(), 100 * A[:, 3].mean(), 100 * A[:, 3].std(ddof=1)))
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'seedtable.tex'), 'w').write('\n'.join(L))

def etae_numbers():
    e = ld('rev_etae.npz')
    if e is None: return
    NREP = list(e['NREP']); i6, i7 = NREP.index(1e6), NREP.index(1e7); tk = ['narma', 'lorenz', 'nce', 'laser']
    L = [r'\begin{tabular}{l' + 'cc' * len(tk) + 'c}', r'\toprule', r'$\eta_e$ & ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % NM[n].replace('–', '--') for n in tk) + r' & passes for $\sigma_{\rm rel}=1\%$\\',
         ' & ' + ' & '.join([r'$10^6$ & $10^7$'] * len(tk)) + r' & (relative to $\eta_e=0.5$)\\', r'\midrule']
    for eta in (0.5, 0.3, 0.1):
        rat = [(e[f'{n}_{k}_eta{eta}'][i6, 1] / e[f'{n}_{k}_eta0.5'][i6, 1]) ** 2 for n in T for k in ('base', 'sq')]
        m('etaPass' + {0.5: 'Five', 0.3: 'Three', 0.1: 'One'}[eta], '%.1f' % np.median(rat)); m('etaPassMax' + {0.5: 'Five', 0.3: 'Three', 0.1: 'One'}[eta], '%.1f' % np.max(rat))
        row = []
        for n in tk:
            rb, rs = e[f'{n}_base_eta{eta}'], e[f'{n}_sq_eta{eta}']
            g6, g7 = pc(rs[i6, 8], rb[i6, 8]), pc(rs[i7, 8], rb[i7, 8]); row += ['%.0f' % g6, '%.0f' % g7]
            m('etaGainSix' + {0.5: 'Five', 0.3: 'Three', 0.1: 'One'}[eta] + U[n], '%.0f' % g6); m('etaGainSeven' + {0.5: 'Five', 0.3: 'Three', 0.1: 'One'}[eta] + U[n], '%.0f' % g7)
        L.append('%.1f & ' % eta + ' & '.join(row) + r' & %.1f\\' % np.median(rat))
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'etatable.tex'), 'w').write('\n'.join(L))

def sme_numbers():
    s = ld('rev_sme.npz')
    if s is None or 'sq_ratio' not in s.files: return
    rows = []; allr = []
    for k in ('base', 'sq'):
        for q in ('Q', 'P'):
            tag = f'{k}_{q}_dt0.005'; r = {}
            for c in ('vQ', 'vQ2', 'cQ', 'vs'):
                tv, mv = s[f'{tag}_traj_{c}'], s[f'{tag}_modv_{c}']; ms = s[f'{tag}_mods_{c}']
                r[c] = (np.median(tv / mv), np.percentile(tv / mv, 10), np.percentile(tv / mv, 90), np.median(tv / ms))
            rows.append((k, q, r)); allr += [r['vQ'][0], r['vQ2'][0], r['vs'][0]]
    for k, q, r in rows:
        kk = ('Base' if k == 'base' else 'Sq') + q
        for c in ('vQ', 'vQ2', 'vs'): m('sme' + kk + {'vQ': 'V', 'vQ2': 'VV', 'vs': 'S'}[c], '%.2f' % r[c][0])
        m('sme' + kk + 'VSq', '%.2f' % r['vQ'][3]); m('sme' + kk + 'VVSq', '%.2f' % r['vQ2'][3])
    dts = []
    for k in ('base', 'sq'):
        for c in ('vQ', 'vQ2', 'vs'):
            dtc = [np.median(s[f'{k}_Q_dt{d}_traj_{c}'] / s[f'{k}_Q_dt{d}_modv_{c}']) for d in (0.005, 0.0025)]; dts.append(100 * abs(dtc[0] - dtc[1]) / dtc[1])
    m('smeDt', '%.1f' % max(dts))
    m('smeMin', '%.2f' % min(allr)); m('smeMax', '%.2f' % max(allr))
    for k in ('base', 'sq'):
        lo = s[f'{k}_loss']; L0 = s[f'{k}_L0']; kk = 'Base' if k == 'base' else 'Sq'
        m('smeBin' + kk, '%+.1f' % (100 * (L0[1] / L0[0] - 1)))
        for i, N in enumerate(lo[:, 0]):
            w = W[round(np.log10(N))]
            m('smeSig' + kk + w, '%.2f' % (lo[i, 4] / lo[i, 2])); m('smeMean' + kk + w, '%+.1f' % (100 * (lo[i, 3] / lo[i, 1] - 1)))
    # measured squeezing gain with model vs trajectory-calibrated covariances
    lb, ls = s['base_loss'], s['sq_loss']
    for i, N in enumerate(lb[:, 0]):
        w = W[round(np.log10(N))]; m('smeGainModel' + w, '%.0f' % pc(ls[i, 1], lb[i, 1])); m('smeGainTraj' + w, '%.0f' % pc(ls[i, 3], lb[i, 3]))
    L = [r'\begin{tabular}{llcccc}', r'\toprule', r'point & setting & $\mathrm{Var}[\hat X]$ & $\mathrm{Var}[\widehat{X^2}]$ & $\mathrm{Cov}[\hat X,\widehat{X^2}]$ & $\mathrm{Var}[\hat\sigma]$\\', r'\midrule']
    for k, q, r in rows:
        L.append(('tuned base' if k == 'base' else 'squeezed optimum') + ' & $(%s,\\sigma_%s)$ & ' % (q, 'x' if q == 'Q' else 'y') + ' & '.join(
            '%.2f [%.2f, %.2f]' % r[c][:3] for c in ('vQ', 'vQ2', 'cQ', 'vs')) + r'\\')
        if k == 'sq':
            L.append(r' & \quad input-noise model & %.2f & %.2f & %.2f & %.2f\\' % tuple(r[c][3] for c in ('vQ', 'vQ2', 'cQ', 'vs')))
    L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(PAPER, 'smetable.tex'), 'w').write('\n'.join(L))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.4))
    for ax, c, t in zip(axs, ('vQ', 'vQ2', 'vs'), (r'$\mathrm{Var}[\hat X]$', r'$\mathrm{Var}[\widehat{X^2}]$', r'$\mathrm{Var}[\hat\sigma]$')):
        for k, mk in (('base', 'o'), ('sq', 's')):
            for q, col in (('Q', 'C0'), ('P', 'C3')):
                tag = f'{k}_{q}_dt0.005'; mv, tv = s[f'{tag}_modv_{c}'], s[f'{tag}_traj_{c}']
                ax.plot(mv, tv, mk, ms=2.5, mfc='none' if k == 'base' else col, color=col, lw=0, label=('tuned base' if k == 'base' else 'squeezed') + ', ' + q)
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]; ax.plot(lim, lim, 'k-', lw=0.5)
        ax.set_xlabel('bin model (vacuum $s_c$)'); ax.set_ylabel('trajectories'); ax.set_title(t, fontsize=8)
    axs[0].legend(frameon=False, fontsize=5)
    fig.savefig(os.path.join(FIG, 'figS_sme.pdf'), bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    refabsq_numbers(); rows = fig_refab(); numbers_refab(rows)
    if ld('rev_grad.npz') is not None and ld('rev_train.npz') is not None:
        gr_, trn_, task_ = fig_cost(); numbers_cost(gr_, ld('rev_train.npz'), task_)
    if ld('rev_nonideal.npz') is not None: fig_nonideal()
    perf_and_eff_tables()
    if ld('rev_asr2.npz') is not None and 'wer5' in ld('rev_asr2.npz').files: asr_numbers()
    if ld('rev_snr.npz') is not None: snr_numbers()
    if ld('rev_train2.npz') is not None: train2_numbers()
    if ld('rev_lo.npz') is not None and ld('rev_lo_diag.npz') is not None: lo_numbers()
    R = bounds_rows()
    if R: bounds_numbers(R); bounds_table(R); fig_bounds(R)
    seeds_numbers(); etae_numbers(); sme_numbers()
    with open(os.path.join(PAPER, 'revnumbers.tex'), 'w') as f:
        for k, v in M.items(): f.write('\\newcommand{\\%s}{%s}\n' % (k, v))
    print('\n'.join('%s = %s' % kv for kv in M.items()))
